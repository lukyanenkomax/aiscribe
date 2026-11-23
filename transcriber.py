"""Потоковая транскрибация аудио чанками с чекпоинтами.

Ключевые свойства для многочасовых записей:
- аудио читается из WAV блоками (не загружается целиком в память);
- результат каждого чанка сразу дописывается в segments.jsonl;
- после каждого сегмента обновляется позиция next_offset в state.json,
  поэтому прогресс не теряется при падении/перезапуске сервиса;
- при возобновлении обработка начинается с последней сохранённой позиции.
"""

import json
import os
import threading
from pathlib import Path
from typing import Callable, Optional

# В anaconda/OpenBLAS-MKL и ctranslate2/onnxruntime лежат разные рантаймы OpenMP;
# без этого флага второй экземпляр роняет процесс с "OMP: Error #15".
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

CHUNK_SEC = 300  # длина чанка (сек): гранулярность чекпоинта и границы VAD

_model_lock = threading.Lock()
_models: dict = {}


class JobCanceled(Exception):
    pass


def get_model(model_size: str, compute_type: str) -> WhisperModel:
    """Кэш моделей: на одном процессе модель грузится один раз."""
    key = (model_size, compute_type)
    with _model_lock:
        if key not in _models:
            _models[key] = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        return _models[key]


def _append_segment(f, seg: dict) -> None:
    f.write(json.dumps(seg, ensure_ascii=False) + "\n")
    f.flush()


def load_segments(job_dir: Path) -> list[dict]:
    path = job_dir / "segments.jsonl"
    if not path.exists():
        return []
    segs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                segs.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # оборванная последняя строка после сбоя — пропускаем
    return segs


def _fmt_ts(seconds: float, sep: str = ".") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def export_outputs(job_dir: Path) -> dict:
    """Собрать transcript.txt / .srt / .vtt из segments.jsonl."""
    segs = load_segments(job_dir)
    txt = "\n".join(s["text"].strip() for s in segs)

    def blocks(ext: str) -> str:
        out = []
        if ext == "vtt":
            out.append("WEBVTT\n")
        for i, s in enumerate(segs, 1):
            if ext == "srt":
                out.append(str(i))
            sep = "," if ext in ("srt", "vtt") else "."
            out.append(f"{_fmt_ts(s['start'], sep)} --> {_fmt_ts(s['end'], sep)}")
            out.append(s["text"].strip())
            out.append("")
        return "\n".join(out)

    paths = {}
    for name, content in (
        ("transcript.txt", txt),
        ("transcript.srt", blocks("srt")),
        ("transcript.vtt", blocks("vtt")),
    ):
        p = job_dir / name
        p.write_text(content, encoding="utf-8")
        paths[name] = p
    return paths


def transcribe(
    job_dir: Path,
    state: dict,
    settings: dict,
    update_state: Callable[[dict], None],
    cancel_event: threading.Event,
) -> None:
    """Транскрибация audio.wav из job_dir с возобновлением с позиции state['next_offset'].

    update_state(partial) — колбэк менеджера задач: сохранить частичное состояние.
    """
    wav = job_dir / "audio.wav"
    model = get_model(settings["model"], settings["compute_type"])
    language = settings.get("language") or None

    start_offset = float(state.get("next_offset") or 0.0)
    detected = state.get("language")

    with sf.SoundFile(str(wav)) as f, open(job_dir / "segments.jsonl", "a", encoding="utf-8") as sf_out:
        offset = 0.0
        # пропуск уже обработанных чанков без декодирования
        if start_offset > 0:
            f.seek(int(start_offset * f.samplerate))
            offset = start_offset

        while True:
            if cancel_event.is_set():
                raise JobCanceled()

            frames = int(CHUNK_SEC * f.samplerate)
            data = f.read(frames, dtype="int16", always_2d=False)
            if data is None or len(data) == 0:
                break
            if data.ndim > 1:
                data = data.mean(axis=1)
            audio = data.astype(np.float32) / 32768.0

            seg_iter, info = model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=settings.get("vad", True),
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False,
            )
            if language is None and info.language and not detected:
                detected = info.language
                update_state({"language": detected})

            chunk_end = offset + len(audio) / f.samplerate
            for seg in seg_iter:
                if cancel_event.is_set():
                    raise JobCanceled()
                item = {
                    "start": round(offset + seg.start, 3),
                    "end": round(offset + seg.end, 3),
                    "text": seg.text.strip(),
                }
                _append_segment(sf_out, item)
                update_state({
                    "next_offset": round(offset + seg.end, 3),
                    "progress": min(0.999, (offset + seg.end) / state["duration"]),
                })

            offset = chunk_end
            update_state({
                "next_offset": round(offset, 3),
                "progress": min(0.999, offset / state["duration"]),
            })

    update_state({"next_offset": state["duration"], "progress": 1.0})