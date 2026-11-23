"""Менеджер фоновых задач: очередь + один рабочий поток.

Состояние каждой задачи атомарно сохраняется в jobs/<id>/state.json,
результат — в jobs/<id>/segments.jsonl. Поэтому:
- Streamlit-интерфейс не зависает (работа идёт в фоновом потоке);
- прогресс переживает перезапуск сервиса (прерванные задачи можно «Возобновить»).
"""

import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import downloader
import summarizer
import transcriber

ROOT = Path(__file__).parent / "jobs"

TERMINAL = {"done", "error", "canceled"}
RESUMABLE_AUDIO = {"done"}  # статусы, при которых audio.wav уже есть

_tmp_lock = threading.RLock()


def _atomic_write(path: Path, data: dict) -> None:
    import json
    tmp = path.with_suffix(".tmp")
    with _tmp_lock:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def _read_state(path: Path) -> dict:
    import json
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


class JobManager:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self.root.mkdir(exist_ok=True)
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.jobs: dict[str, dict] = {}
        self.cancel_events: dict[str, threading.Event] = {}
        self._sum_keys: dict[str, str] = {}  # ключ для авто-изложения (только в памяти)
        self._load_existing()
        self._worker = threading.Thread(target=self._run_loop, daemon=True, name="transcribe-worker")
        self._worker.start()

    # ---------- состояние ----------

    def _load_existing(self) -> None:
        need_meta = []
        for d in sorted(self.root.iterdir()):
            st = d / "state.json"
            if not st.exists():
                continue
            state = _read_state(st)
            if not state:
                continue
            changed = False
            if state.get("status") not in TERMINAL:
                state["status"] = "interrupted"
                state["stage"] = "Прервано перезапуском сервиса — нажмите «Возобновить»"
                changed = True
            if state.get("summary_status") in ("running", "pending"):
                state["summary_status"] = "error"
                state["summary_error"] = "Прервано перезапуском сервиса — повторите запрос"
                changed = True
            # метаданные ролика (название/канал), если уже скачивались
            meta = downloader.load_meta(d)
            if meta:
                patch = self._meta_patch(state, meta)
                if patch:
                    state.update(patch)
                    changed = True
            if changed:
                _atomic_write(st, state)
            self.jobs[state["id"]] = state
            if (state.get("source_type") == "url" and state.get("source", "").startswith("http")
                    and not state.get("channel")):
                need_meta.append(state["id"])
        if need_meta:
            threading.Thread(target=self._fetch_meta, args=(need_meta,),
                             daemon=True, name="meta-fetch").start()

    @staticmethod
    def _meta_patch(state: dict, meta: dict) -> dict:
        """Что обновить в состоянии задачи из метаданных ролика."""
        patch = {}
        if meta.get("channel") and not state.get("channel"):
            patch["channel"] = meta["channel"]
        if meta.get("title") and (not state.get("title")
                                  or state.get("title") == state.get("source")):
            patch["title"] = meta["title"]
        if meta.get("duration") and not state.get("duration"):
            patch["duration"] = float(meta["duration"])
        return patch

    def _fetch_meta(self, job_ids: list) -> None:
        """Дозапросить метаданные для уже существующих url-задач (фоном)."""
        for jid in job_ids:
            job = self.jobs.get(jid)
            if not job or jid not in self.jobs:
                continue
            url = job.get("source", "")
            try:
                info = downloader.fetch_info(url)
                meta = downloader.meta_from_info(info)
            except Exception:
                continue  # недоступность ссылки — не критично, карточка останется без канала
            downloader.save_meta(self.root / jid, meta)
            patch = self._meta_patch(self.jobs[jid], meta)
            if patch:
                self.update(jid, patch)
            time.sleep(1)  # вежливость к серверу

    def _persist(self, job_id: str) -> None:
        _atomic_write(self.root / job_id / "state.json", self.jobs[job_id])

    def update(self, job_id: str, patch: dict) -> None:
        with _tmp_lock:
            self.jobs[job_id].update(patch)
            state = dict(self.jobs[job_id])
        _atomic_write(self.root / job_id / "state.json", state)

    def list(self) -> list[dict]:
        return sorted(self.jobs.values(), key=lambda j: j.get("created_at", 0), reverse=True)

    def get(self, job_id: str) -> dict:
        return self.jobs[job_id]

    # ---------- создание / управление ----------

    def create_job(self, *, source_type: str, source: str = "", title: str = "",
                   settings: dict, auto_summary: dict | None = None,
                   summary_key: str = "") -> str:
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "id": job_id,
            "source_type": source_type,   # url | file | path
            "source": source,              # url или исходный путь
            "title": title or source or job_id,
            "settings": settings,
            "status": "queued",
            "stage": "В очереди",
            "progress": 0.0,
            "next_offset": 0.0,
            "duration": 0.0,
            "language": None,
            "error": None,
            "auto_summary": auto_summary,  # {"model","lang"} — ключ сюда не пишем
            "created_at": time.time(),
        }
        self.jobs[job_id] = state
        if summary_key:
            self._sum_keys[job_id] = summary_key
        self._persist(job_id)
        self.enqueue(job_id)
        return job_id

    def enqueue(self, job_id: str) -> None:
        self.cancel_events[job_id] = threading.Event()
        self.update(job_id, {"status": "queued", "stage": "В очереди", "error": None})
        self.queue.put(job_id)

    def cancel(self, job_id: str) -> None:
        ev = self.cancel_events.get(job_id)
        if ev:
            ev.set()
        if self.jobs[job_id]["status"] in ("queued", "interrupted", "error"):
            self.update(job_id, {"status": "canceled", "stage": "Отменено"})

    def delete(self, job_id: str) -> None:
        self.cancel(job_id)
        self.jobs.pop(job_id, None)
        self._sum_keys.pop(job_id, None)
        shutil.rmtree(self.root / job_id, ignore_errors=True)

    def set_summary_key(self, job_id: str, key: str) -> None:
        """Запомнить ключ авто-изложения (в памяти) — например, после рестарта."""
        if key and self.jobs.get(job_id, {}).get("auto_summary"):
            self._sum_keys[job_id] = key

    # ---------- изложение (LLM) ----------

    def start_summary(self, job_id: str, api_key: str, model: str,
                      lang: str = "ru") -> None:
        if self.jobs[job_id].get("summary_status") == "running":
            return
        self.update(job_id, {
            "summary_status": "running",
            "summary_stage": "Подготовка",
            "summary_progress": 0.0,
            "summary_error": None,
        })
        threading.Thread(target=self._run_summary, args=(job_id, api_key, model, lang),
                         daemon=True, name=f"summary-{job_id}").start()

    def _run_summary(self, job_id: str, api_key: str, model: str,
                     lang: str = "ru") -> None:
        job_dir = self.root / job_id
        try:
            txt = job_dir / "transcript.txt"
            if not txt.exists():
                transcriber.export_outputs(job_dir)
            text = txt.read_text(encoding="utf-8")

            def prog(frac, msg):
                self.update(job_id, {"summary_progress": frac, "summary_stage": msg})

            out = summarizer.summarize(text, api_key, model, progress_cb=prog, lang=lang)
            (job_dir / "summary.md").write_text(out, encoding="utf-8")
            self.update(job_id, {"summary_status": "done", "summary_stage": "Готово",
                                 "summary_progress": 1.0})
        except Exception:
            self.update(job_id, {
                "summary_status": "error",
                "summary_stage": "Ошибка",
                "summary_error": traceback.format_exc(limit=5),
            })

    # ---------- воркер ----------

    def _run_loop(self) -> None:
        while True:
            job_id = self.queue.get()
            if job_id not in self.jobs:
                continue
            try:
                self._run(job_id)
            except Exception:
                self.update(job_id, {
                    "status": "error",
                    "stage": "Ошибка",
                    "error": traceback.format_exc(limit=5),
                })

    def _run(self, job_id: str) -> None:
        job = self.jobs[job_id]
        if job.get("status") == "canceled":  # отменена, пока стояла в очереди
            return
        job_dir = self.root / job_id
        settings = job["settings"]
        cancel = self.cancel_events[job_id]

        def check_cancel():
            if cancel.is_set():
                raise transcriber.JobCanceled()

        # 1. Готовые субтитры (опция): пробуем их вместо аудио и транскрибации
        wav = job_dir / "audio.wav"
        if (job["source_type"] == "url" and settings.get("use_subs")
                and not wav.exists() and not (job_dir / "segments.jsonl").exists()):
            self.update(job_id, {"status": "downloading", "progress": 0.05,
                                 "stage": "Поиск готовых субтитров"})

            def subs_progress(frac, msg):
                check_cancel()
                self.update(job_id, {"stage": msg})

            got = downloader.fetch_subtitles(job["source"], job_dir,
                                             progress_cb=subs_progress)
            check_cancel()
            if got:
                downloader.save_meta(job_dir, {k: got.get(k, "") for k in
                                               ("title", "channel", "duration",
                                                "webpage_url")})
                with open(job_dir / "segments.jsonl", "w", encoding="utf-8") as f:
                    for seg in got["segments"]:
                        f.write(json.dumps(seg, ensure_ascii=False) + "\n")
                self.update(job_id, {
                    **self._meta_patch(self.jobs[job_id], got),
                    "language": got.get("language") or None,
                    "stage": "Экспорт TXT/SRT/VTT",
                })
                transcriber.export_outputs(job_dir)
                self.update(job_id, {"status": "done", "stage": "Готово (субтитры)",
                                     "progress": 1.0, "stage_progress": None,
                                     "finished_at": time.time()})
                self._auto_summary(job_id)
                return
            # субтитров нет или не скачались — обычная загрузка аудио
            self.update(job_id, {"stage": "Субтитры не найдены — загрузка аудио",
                                 "progress": 0.0, "stage_progress": None})

        # 2. Аудио: загрузка по ссылке / подготовка локального файла
        if not wav.exists():
            self.update(job_id, {"status": "downloading", "stage": "Получение аудио", "progress": 0.0})

            def dl_progress(frac, msg):
                check_cancel()
                patch = {"stage": msg}
                if frac is not None:
                    patch["stage_progress"] = frac
                self.update(job_id, patch)

            def got_meta(meta):
                # данные о ролике получены — сразу показываем в карточке
                self.update(job_id, self._meta_patch(self.jobs[job_id], meta))

            if job["source_type"] == "url":
                downloader.download_url(job["source"], job_dir,
                                        progress_cb=dl_progress, meta_cb=got_meta)
            else:
                downloader.prepare_local_file(Path(job["source"]), job_dir, progress_cb=dl_progress)
            self.update(job_id, {"stage_progress": None})

        check_cancel()

        # 2. Длительность (для прогресса) и старт/возобновление транскрибации
        if not job.get("duration"):
            duration = downloader.probe_duration(wav)
            self.update(job_id, {"duration": duration})
        if job["duration"] <= 0:
            self.update(job_id, {"status": "error", "stage": "Ошибка",
                                  "error": "Не удалось определить длительность аудио"})
            return

        started = time.time()
        processed0 = float(job.get("next_offset") or 0.0)
        last_tick = [time.time()]

        def upd(partial: dict):
            # heartbeat: скорость и оценка оставшегося времени
            now = time.time()
            if now - last_tick[0] > 2:
                done = float(partial.get("next_offset") or self.jobs[job_id]["next_offset"])
                rate = (done - processed0) / max(now - started, 1e-6)  # сек аудио / сек реального времени
                partial["audio_per_real"] = round(rate, 3)
                last_tick[0] = now
            self.update(job_id, partial)

        self.update(job_id, {
            "status": "transcribing",
            "stage": "Транскрибация" if not job.get("next_offset") else "Транскрибация (продолжение)",
        })

        try:
            transcriber.transcribe(job_dir, self.jobs[job_id], settings, upd, cancel)
        except transcriber.JobCanceled:
            self.update(job_id, {"status": "canceled", "stage": "Отменено пользователем"})
            return

        check_cancel()

        # 3. Экспорт результатов
        self.update(job_id, {"stage": "Экспорт TXT/SRT/VTT"})
        transcriber.export_outputs(job_dir)
        self.update(job_id, {"status": "done", "stage": "Готово", "progress": 1.0,
                             "finished_at": time.time()})

        # 4. Авто-изложение (опция «сделать изложение после транскрибации»)
        self._auto_summary(job_id)

    def _auto_summary(self, job_id: str) -> None:
        """Запустить изложение после успешной транскрибации, если включена опция."""
        job = self.jobs.get(job_id)
        auto = job.get("auto_summary") if job else None
        key = self._sum_keys.get(job_id)
        if auto and key and not (self.root / job_id / "summary.md").exists():
            self.start_summary(job_id, key, auto.get("model", "openrouter/free"),
                               auto.get("lang", "ru"))


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager