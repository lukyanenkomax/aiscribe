"""Получение аудио: загрузка по ссылке (yt-dlp) или конвертация локального файла
в 16 kHz моно WAV (единый формат для транскрибации, стриминговое чтение)."""

import html
import json
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yt_dlp

_IS_WIN = subprocess.os.name == "nt"
_NO_WINDOW = {"creationflags": 0x08000000} if _IS_WIN else {}
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


class DownloadError(Exception):
    pass


def meta_from_info(info: dict) -> dict:
    """Метаданные ролика для карточки задачи (название, канал, длительность)."""
    return {
        "title": info.get("title") or "",
        "channel": info.get("channel") or info.get("uploader") or "",
        "duration": float(info.get("duration") or 0.0),
        "webpage_url": info.get("webpage_url") or "",
    }


def save_meta(workdir: Path, meta: dict) -> None:
    (workdir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_meta(workdir: Path) -> dict:
    try:
        return json.loads((workdir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def fetch_info(url: str) -> dict:
    """Только метаданные по ссылке, без скачивания аудио."""
    if _is_castbox(url):
        return parse_castbox(_castbox_page_url(url))
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info.get("_type") == "playlist":
        entries = info.get("entries") or []
        info = entries[0] if entries else info
    return info


# ---------- Готовые субтитры (YouTube и другие площадки yt-dlp) ----------

_TS = r"(?:\d{1,2}:)?\d{1,2}:\d{1,2}[.,]\d{1,3}"


def _vtt_seconds(ts: str) -> float:
    parts = ts.replace(",", ".").split(":")
    if len(parts) == 2:
        parts = ["0"] + parts
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _overlap_cut(prev: str, cur: str) -> int | None:
    """Сколько символов в начале cur повторяют конец prev (пословно).

    Автосубтитры YouTube идут «бегущей строкой»: каждый cue повторяет хвост
    предыдущего и добавляет новый текст. Возвращает индекс в cur, с которого
    начинается новое, или None, если перекрытия нет.
    """
    prev_words = [w.group(0).lower() for w in _WORD_RE.finditer(prev)]
    cur_matches = list(_WORD_RE.finditer(cur))
    cur_words = [m.group(0).lower() for m in cur_matches]
    limit = min(len(prev_words), len(cur_words))
    for k in range(limit, 0, -1):
        if prev_words[-k:] == cur_words[:k] and (k >= 2 or k == len(cur_words)):
            return cur_matches[k].start() if k < len(cur_matches) else len(cur)
    return None


def strip_rolling_duplicates(segments: list[dict]) -> list[dict]:
    """Убирает повторы «бегущей строки» в сегментах, оставляя только новое."""
    out: list[dict] = []
    prev = ""
    for seg in segments:
        text = seg["text"]
        cut = _overlap_cut(prev, text)
        if cut is None:
            prev = text
            out.append(seg)
            continue
        rest = text[cut:].strip()
        if rest:
            out.append({**seg, "text": rest})
            prev = text
    return out


_SENT_END = (".", "!", "?", "…", ":")


def merge_segments(segments: list[dict], max_chars: int = 140,
                   max_gap: float = 1.5) -> list[dict]:
    """Склеивает короткие обрывки в читаемые строки (в пределах предложения)."""
    out: list[dict] = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if out:
            last = out[-1]
            gap = seg["start"] - last["end"]
            if (gap <= max_gap and not last["text"].endswith(_SENT_END)
                    and len(last["text"]) + len(text) + 1 <= max_chars):
                last["text"] = f"{last['text']} {text}"
                last["end"] = seg["end"]
                continue
        out.append({"start": seg["start"], "end": seg["end"], "text": text})
    return out


def parse_subtitles(path: Path) -> list[dict]:
    """VTT/SRT -> сегменты {start, end, text} в формате segments.jsonl.

    Убирает теги разметки (<c>, тайм-коды слов) и повторы «бегущей строкой»,
    которыми YouTube отдаёт автосубтитры.
    """
    segments: list[dict] = []
    cur_ts, cur_lines = None, []

    def flush() -> None:
        if cur_ts and cur_lines:
            text = " ".join(cur_lines).strip()
            if text:
                segments.append({"start": round(cur_ts[0], 3),
                                 "end": round(cur_ts[1], 3), "text": text})

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if "-->" in line:
            flush()
            m = re.match(rf"({_TS})\s*-->\s*({_TS})", line)
            cur_ts = (_vtt_seconds(m.group(1)), _vtt_seconds(m.group(2))) if m else None
            cur_lines = []
        elif cur_ts is not None and line:
            text = html.unescape(re.sub(r"<[^>]+>", "", line)).strip()
            if text:
                cur_lines.append(text)
    flush()
    return merge_segments(strip_rolling_duplicates(segments))


def fetch_subtitles(url: str, workdir: Path, progress_cb=None) -> dict | None:
    """Готовые субтитры (обычные или автосубтитры) вместо локальной транскрибации.

    Возвращает {"segments", "title", "channel", "duration", "language", ...}
    или None, если субтитров нет / скачать не удалось — тогда по-прежнему
    работает обычная загрузка аудио и транскрибация.
    """
    if progress_cb:
        progress_cb(None, "Поиск готовых субтитров...")

    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return None
    if info.get("_type") == "playlist":
        entries = info.get("entries") or []
        info = entries[0] if entries else info

    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    if not manual and not auto:
        return None

    # кандидаты: язык оригинала (полный код и базовый), затем русский
    orig = (info.get("language") or "").strip()
    candidates: list[str] = []
    for lang in (orig, orig.split("-")[0] if orig else "", "ru"):
        if lang and lang not in candidates and (manual.get(lang) or auto.get(lang)):
            candidates.append(lang)
    if not candidates:
        candidates = list(manual) or list(auto)

    for lang in candidates:
        # запрашиваем ровно один язык: yt-dlp падает целиком, если не скачать
        # хотя бы один из запрошенных (например, HTTP 429 на timedtext).
        # YouTube периодически отдаёт 429 — ждём и повторяем, прежде чем
        # сдаться и перейти к обычной транскрибации.
        dl_opts = {**opts, "writesubtitles": True, "writeautomaticsub": True,
                   "subtitleslangs": [lang], "subtitlesformat": "vtt/best",
                   "outtmpl": str(workdir / "_subs.%(ext)s")}
        for attempt in range(3):
            if progress_cb:
                msg = f"Субтитры: загрузка ({lang})..."
                if attempt:
                    msg = (f"Субтитры: YouTube ограничил запросы (429) — "
                           f"повтор через {10 * attempt} с…")
                progress_cb(None, msg)
            if attempt:
                time.sleep(10 * attempt)
            try:
                with yt_dlp.YoutubeDL(dl_opts) as ydl:
                    ydl.extract_info(url, download=True)
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    continue  # подождём и повторим
                break  # субтитров нет / прочая ошибка — следующий язык
        files = [p for p in workdir.glob("_subs.*") if p.suffix in (".vtt", ".srt")]
        if not files:
            continue
        segments = parse_subtitles(files[0])
        for p in workdir.glob("_subs.*"):
            p.unlink(missing_ok=True)
        if segments:
            return {
                "segments": segments,
                "title": info.get("title") or "",
                "channel": info.get("channel") or info.get("uploader") or "",
                "duration": info.get("duration") or 0,
                "language": lang,
                "webpage_url": info.get("webpage_url") or url,
            }
    return None


# ---------- Castbox (yt-dlp не поддерживает castbox.fm) ----------

def _is_castbox(url: str) -> bool:
    return "castbox.fm" in url.lower()


def _castbox_page_url(url: str) -> str:
    """Привести ссылку castbox к странице эпизода (/ep/ID или /vb/ID -> /ep/ID)."""
    m = re.search(r"castbox\.fm/(?:ep|vb)/(\w+)", url, re.I)
    if m:
        return f"https://castbox.fm/ep/{m.group(1)}"
    return url


def parse_castbox(page_url: str) -> dict:
    """Метаданные эпизода + прямая ссылка на аудио со страницы castbox."""
    req = urllib.request.Request(page_url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            page = resp.read().decode("utf-8", "replace")
    except Exception as e:
        raise DownloadError(f"Castbox: не удалось открыть страницу эпизода: {e}") from e

    m = re.search(r'https?://[^"\'\s\\]+\.mp3[^"\'\s\\]*', page)
    if not m:
        raise DownloadError("Castbox: на странице не найдена ссылка на аудио "
                            "(возможно, эпизод удалён или требуется другой регион)")
    audio_url = html.unescape(m.group(0))

    title = ""
    og = re.search(r'<meta property="og:title" content="([^"]*)"', page)
    if og:
        title = html.unescape(og.group(1)).strip()

    channel = ""
    ld = re.search(r'<script type="application/ld\+json">(.*?)</script>', page, re.S)
    if ld:
        try:
            items = json.loads(ld.group(1)).get("itemListElement") or []
            names = [urllib.parse.unquote(it.get("item", {}).get("name", "")) for it in items]
            # порядок: [Channels, <канал>, <эпизод>]
            if len(names) >= 2:
                channel = names[1]
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    return {
        "title": title,
        "channel": channel,
        "duration": 0.0,
        "webpage_url": page_url,
        "audio_url": audio_url,
    }


def _download_direct(audio_url: str, workdir: Path, progress_cb=None) -> Path:
    """Прямая загрузка аудио по ссылке (с прогрессом) в workdir/source.mp3."""
    dst = workdir / "source.mp3"
    req = urllib.request.Request(audio_url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(dst, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress_cb:
                    if total:
                        progress_cb(done / total,
                                    f"Загрузка аудио: {done / 1048576:.1f} / {total / 1048576:.1f} МБ")
                    else:
                        progress_cb(None, f"Загрузка аудио: {done / 1048576:.1f} МБ")
    except Exception as e:
        dst.unlink(missing_ok=True)
        raise DownloadError(f"Castbox: не удалось скачать аудио: {e}") from e
    if not dst.exists() or dst.stat().st_size == 0:
        raise DownloadError("Castbox: скачан пустой аудиофайл")
    return dst


def download_castbox(url: str, workdir: Path, progress_cb=None, meta_cb=None) -> Path:
    """Загрузка эпизода castbox: страница -> ссылка на аудио -> WAV."""
    if progress_cb:
        progress_cb(None, "Castbox: получение страницы эпизода...")
    info = parse_castbox(_castbox_page_url(url))
    meta = {k: info.get(k, "") for k in ("title", "channel", "duration", "webpage_url")}
    save_meta(workdir, meta)
    if meta_cb:
        try:
            meta_cb(meta)
        except Exception:
            pass
    if progress_cb:
        progress_cb(None, "Castbox: загрузка аудио...")
    src = _download_direct(info["audio_url"], workdir, progress_cb)
    if progress_cb:
        progress_cb(None, "Конвертация в WAV...")
    wav = convert_to_wav(src, workdir / "audio.wav")
    src.unlink(missing_ok=True)
    return wav


def _ffmpeg_exe() -> str:
    """ffmpeg из PATH, иначе бинарь из пакета imageio-ffmpeg (ставится с pip)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def probe_duration(path: Path) -> float:
    """Длительность файла в секундах (soundfile, затем ffprobe)."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        if info.samplerate:
            return float(info.frames) / info.samplerate
    except Exception:
        pass
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, **_NO_WINDOW)
    except FileNotFoundError:
        raise DownloadError(
            "Не найдены ffmpeg/ffprobe. Установите ffmpeg (см. README) или "
            "переустановите зависимости: pip install -r requirements.txt") from None
    if out.returncode != 0:
        raise DownloadError(f"ffprobe не смог прочитать файл: {out.stderr.strip()[:300]}")
    try:
        return float(json.loads(out.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise DownloadError(f"Не удалось определить длительность: {e}")


def convert_to_wav(src: Path, dst: Path) -> Path:
    """Любой формат/контейнер -> 16 kHz моно PCM WAV."""
    cmd = [
        _ffmpeg_exe(), "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        str(dst),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, **_NO_WINDOW)
    except FileNotFoundError:
        raise DownloadError(
            "Не найден ffmpeg. Установите ffmpeg (см. README) или переустановите "
            "зависимости: pip install -r requirements.txt") from None
    if out.returncode != 0 or not dst.exists():
        raise DownloadError(f"ffmpeg не смог конвертировать аудио: {out.stderr.strip()[-500:]}")
    return dst


def _ydl_progress_hook(progress_cb):
    def hook(d):
        if progress_cb is None:
            return
        if d.get("status") == "downloading" and d.get("total_bytes") or d.get("total_bytes_estimate"):
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes") or 0
            if total:
                def hz(b):
                    return f"{b / 1048576:.1f} МБ" if b >= 1048576 else f"{b / 1024:.0f} КБ"
                progress_cb(done / total, f"Загрузка аудио: {hz(done)} / {hz(total)}")
        elif d.get("status") == "finished":
            progress_cb(None, "Загрузка завершена, конвертация...")
        elif d.get("status") == "processing":
            progress_cb(None, "Обработка ffmpeg (извлечение аудио)...")
    return hook


def download_url(url: str, workdir: Path, progress_cb=None, meta_cb=None) -> Path:
    """Скачать аудио по ссылке (YouTube и тысячи других сайтов через yt-dlp)
    и конвертировать в audio.wav внутри workdir.

    meta_cb(meta) вызывается сразу после получения данных о ролике
    (название/канал) — ещё до скачивания аудио.
    """
    if _is_castbox(url):
        return download_castbox(url, workdir, progress_cb, meta_cb)

    if progress_cb:
        progress_cb(None, "Получение информации по ссылке...")

    captured = {}

    def _capture(info, incomplete=False):
        # match_filter вызывается yt-dlp сразу после извлечения метаданных,
        # до начала скачивания
        if not incomplete and not captured:
            captured.update(info)
            meta = meta_from_info(info)
            save_meta(workdir, meta)
            if meta_cb:
                try:
                    meta_cb(meta)
                except Exception:
                    pass
        return None

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_ydl_progress_hook(progress_cb)],
        "match_filter": _capture,
        "postprocessors": [],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:  # yt-dlp бросает разные типы ошибок
        raise DownloadError(f"Ошибка загрузки по ссылке: {e}") from e

    if not captured and info:
        # запасной вариант, если match_filter не сработал
        meta = meta_from_info(info)
        save_meta(workdir, meta)
        if meta_cb:
            try:
                meta_cb(meta)
            except Exception:
                pass

    candidates = sorted(workdir.glob("source.*"))
    # исключаем уже готовый wav и служебные файлы
    candidates = [p for p in candidates if p.suffix.lower() != ".part" and not p.name.endswith(".json")]
    if not candidates:
        raise DownloadError("yt-dlp не сохранил аудиофайл")

    wav = workdir / "audio.wav"
    if candidates[0].suffix.lower() == ".wav":
        candidates[0].rename(wav)
    else:
        convert_to_wav(candidates[0], wav)
        candidates[0].unlink(missing_ok=True)
    return wav


def prepare_local_file(src: Path, workdir: Path, progress_cb=None) -> Path:
    """Скопировать/конвертировать локальный файл в workdir/audio.wav."""
    if progress_cb:
        progress_cb(None, "Конвертация локального файла в WAV...")
    wav = workdir / "audio.wav"
    if src.suffix.lower() == ".wav":
        # проверим, что уже 16k mono; иначе конвертируем всё равно — это дёшево
        convert_to_wav(src, wav)
    else:
        convert_to_wav(src, wav)
    return wav