"""Локальный сервис транскрибации: Streamlit UI.

Запуск:  streamlit run app.py
"""

import os
import time
from pathlib import Path

import streamlit as st

import jobs as jobs_mod
import transcriber

st.set_page_config(page_title="Транскрибация аудио", page_icon="🎙️", layout="wide")


@st.cache_resource
def manager() -> jobs_mod.JobManager:
    return jobs_mod.get_manager()


STATUS_ICON = {
    "queued": "⏳", "downloading": "⬇️", "transcribing": "🔄",
    "done": "✅", "error": "⛔", "canceled": "🛑", "interrupted": "⏸️",
}
RESUMABLE = {"interrupted", "error", "canceled"}
RUNNING = {"queued", "downloading", "transcribing"}


def fmt_hms(sec: float) -> str:
    sec = int(sec)
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------- сайдбар: настройки модели ----------------

with st.sidebar:
    st.title("🎙️ Транскрибация")
    st.caption("Локальный сервис: модели и данные не покидают ваш ПК.")
    model = st.selectbox(
        "Модель Whisper", ["tiny", "base", "small", "medium", "large-v3"],
        index=2,
        help="Чем больше — тем точнее и медленнее. Для CPU: small — хороший баланс, "
             "large-v3 только если терпения хватит на многочасовые записи.")
    compute_type = st.selectbox(
        "Тип вычислений", ["int8", "float32"],
        help="int8 — быстрее и компактнее на CPU (рекомендуется).")
    language = st.selectbox(
        "Язык", ["auto", "ru", "en", "de", "es", "fr", "pt", "it", "uk", "pl", "tr", "zh", "ja"],
        help="auto — определится автоматически по началу записи.")
    vad = st.checkbox("VAD-фильтр ( пропуск тишины, ускорение)", value=True)
    use_subs = st.checkbox(
        "📺 Брать готовые субтитры (YouTube)", value=True, key="use_subs",
        help="Если у видео есть субтитры (ручные или автосубтитры) — они будут "
             "использованы напрямую, без загрузки аудио и транскрибации (быстро и "
             "без нагрузки на CPU). Субтитры берутся на языке видео, при отсутствии — "
             "на русском. Если субтитров нет — обычная транскрибация.")
    auto_summary = st.checkbox(
        "📝 Составить изложение после транскрибации", value=True, key="auto_summary",
        help="Сразу после успешного окончания транскрибации изложение будет составлено "
             "автоматически с настройками ниже (модель, ключ, язык). Пока идёт "
             "транскрибация — можно просто ждать, результат появится в карточке.")
    st.divider()
    st.subheader("Изложение (LLM)")
    api_key = st.text_input(
        "API-ключ OpenRouter", type="password", key="or_key",
        value=os.environ.get("OPENROUTER_API_KEY", ""),
        help="Создать: https://openrouter.ai/keys. Ключ хранится только в памяти сессии.")
    llm_model = st.text_input(
        "Модель OpenRouter", value="openrouter/free", key="or_model",
        help="Например: openrouter/free (бесплатная модель по умолчанию) или конкретная, "
             "например deepseek/deepseek-chat-v3.1:free")
    sum_lang = st.selectbox(
        "Язык изложения", ["Русский (перевод)", "Язык оригинала"], key="sum_lang",
        help="Для записи на английском (или любом ином языке): «Русский» — изложение "
             "будет составлено и переведено на русский; «Язык оригинала» — на языке "
             "говорящего.")
    st.divider()
    st.caption(
        "Первый запуск модели скачивает её с Hugging Face (~"
        f"{'0.5' if model in ('tiny', 'base') else '1' if model in ('small',) else '1.5' if model == 'medium' else '3'} "
        "ГБ для medium/large). Задачи выполняются по одной в фоне; "
        "прогресс сохраняется на диск — сервис можно перезапускать.")
    if st.checkbox("Показать папку jobs/", value=False):
        st.code(str(jobs_mod.ROOT))

settings = {
    "model": model,
    "compute_type": compute_type,
    "language": None if language == "auto" else language,
    "vad": vad,
    "use_subs": bool(st.session_state.get("use_subs", True)),
}

# ---------------- создание задачи ----------------

st.subheader("Новая задача")
tab_url, tab_file, tab_path = st.tabs(["🔗 Ссылка (YouTube/подкаст)", "📁 Локальный файл", "⌨️ Путь к файлу"])

with tab_url:
    url = st.text_input(
        "Ссылка на ролик или подкаст",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Поддерживаются YouTube и тысячи других сайтов (yt-dlp): подкасты, SoundCloud, RSS-эпизоды и т.д.")

with tab_file:
    uploaded = st.file_uploader("Аудио/видео файл", type=[
        "mp3", "m4a", "wav", "ogg", "opus", "flac", "aac", "wma", "mp4", "mkv", "webm", "mov", "avi"])

with tab_path:
    local_path = st.text_input("Полный путь к файлу на диске",
                               placeholder=r"C:\Users\you\Documents\podcast.mp3")

if st.button("▶️ Начать транскрибацию", type="primary", use_container_width=True):
    m = manager()
    source_type, source, title = None, "", ""
    if url.strip():
        source_type, source, title = "url", url.strip(), ""
    elif uploaded is not None:
        source_type = "file"
        title = uploaded.name
    elif local_path.strip():
        p = Path(local_path.strip().strip('"'))
        if p.exists():
            source_type, source, title = "path", str(p), p.name
        else:
            st.error(f"Файл не найден: {p}")

    if source_type:
        # авто-изложение после транскрибации (настройки берём из сайдбара сейчас;
        # API-ключ остаётся только в памяти менеджера, на диск не пишется)
        auto_cfg, auto_key = None, ""
        if st.session_state.get("auto_summary"):
            auto_key = st.session_state.get("or_key", "").strip()
            if auto_key:
                auto_cfg = {
                    "model": st.session_state.get("or_model", "openrouter/free"),
                    "lang": "ru" if str(st.session_state.get(
                        "sum_lang", "Русский (перевод)")).startswith("Русский") else "orig",
                }
            else:
                st.warning("«Изложение после транскрибации» включено, но не задан "
                           "API-ключ OpenRouter — изложение составлено не будет.")
        job_id = m.create_job(source_type=source_type, source=source, title=title,
                              settings=settings, auto_summary=auto_cfg,
                              summary_key=auto_key)
        if source_type == "file":
            # сохраняем загруженный файл внутрь задачи, чтобы возобновление работало после рестарта
            dest = m.root / job_id / ("source" + Path(uploaded.name).suffix.lower())
            dest.write_bytes(uploaded.getbuffer())
            m.update(job_id, {"source": str(dest)})
            source = str(dest)
        if source_type == "path" and not Path(source).exists():
            m.delete(job_id)
            st.error("Файл не найден")
        else:
            st.success(f"Задача добавлена в очередь: {title or source}")
    elif not any([url.strip(), uploaded, local_path.strip()]):
        st.warning("Укажите ссылку или файл")

st.divider()

# ---------------- список задач (автообновление) ----------------


@st.fragment(run_every="2s")
def jobs_fragment():
    m = manager()
    all_jobs = m.list()
    if not all_jobs:
        st.info("Задач пока нет. Добавьте ссылку или файл выше.")
        return

    for job in all_jobs:
        jid = job["id"]
        status = job.get("status", "?")
        icon = STATUS_ICON.get(status, "❔")
        dur = job.get("duration") or 0
        title = job.get("title") or job.get("source") or jid
        if len(title) > 80:
            title = title[:77] + "..."
        sub = f"{job.get('settings', {}).get('model', '?')} · {fmt_hms(dur) if dur else 'длительность неизвестна'}"
        if job.get("language"):
            sub += f" · язык: {job['language']}"

        with st.container(border=True):
            c1, c2, c3 = st.columns([6, 2, 2])
            with c1:
                st.markdown(f"**{icon} {title}**")
                channel = (job.get("channel") or "").strip()
                if channel:
                    if len(channel) > 60:
                        channel = channel[:57] + "..."
                    st.caption(f"📺 {channel}")
                st.caption(f"{sub} · ID: `{jid}`")
            with c2:
                st.markdown(f"**{job.get('progress', 0) * 100:.1f}%**")
                if status == "transcribing" and dur and job.get("audio_per_real"):
                    remain = max(dur - job.get("next_offset", 0), 0)
                    st.caption(f"осталось ≈ {fmt_hms(remain / max(job['audio_per_real'], 1e-6))}")
            with c3:
                st.markdown(f"`{status}`")

            stage = job.get("stage") or ""
            if job.get("stage_progress") is not None:
                stage += f" · {job['stage_progress'] * 100:.0f}%"
            st.caption(stage)
            st.progress(min(job.get("progress") or 0.0, 1.0), text="Прогресс транскрибации")

            if job.get("status") == "error" and job.get("error"):
                with st.expander("Показать ошибку"):
                    st.code(job["error"])

            b1, b2, b3, b4, b5 = st.columns([2, 2, 2, 3, 3])
            with b1:
                if status in RESUMABLE:
                    if st.button("▶️ Возобновить", key=f"res-{jid}"):
                        # после рестарта ключ авто-изложения утерян — пробуем восстановить
                        m.set_summary_key(jid, st.session_state.get("or_key", "").strip())
                        m.enqueue(jid)
                        st.rerun(scope="fragment")
            with b2:
                if status in RUNNING:
                    if st.button("🛑 Отменить", key=f"can-{jid}"):
                        m.cancel(jid)
                        st.rerun(scope="fragment")
            with b3:
                if status not in RUNNING:
                    if st.button("🗑️ Удалить", key=f"del-{jid}"):
                        m.delete(jid)
                        st.rerun(scope="fragment")
            if status == "done":
                job_dir = m.root / jid
                transcriber.export_outputs(job_dir)
                d1, d2, d3 = st.columns(3)
                for col, name, label in ((d1, "transcript.txt", "⬇️ TXT"),
                                         (d2, "transcript.srt", "⬇️ SRT"),
                                         (d3, "transcript.vtt", "⬇️ VTT")):
                    with col:
                        p = job_dir / name
                        if p.exists():
                            st.download_button(label, p.read_bytes(), file_name=f"{jid}-{name}",
                                               key=f"dl-{jid}-{name}")
                with st.expander("Предпросмотр текста"):
                    txt = (job_dir / "transcript.txt")
                    if txt.exists():
                        full = txt.read_text(encoding="utf-8")
                        st.caption(f"{len(full):,} символов, {len(full.split()):,} слов"
                                   .replace(",", " "))
                        st.text_area("Полный текст", value=full, height=520,
                                     disabled=True, key=f"prev-{jid}",
                                     label_visibility="collapsed")

                ss = job.get("summary_status")
                sm = job_dir / "summary.md"
                can_sum = bool(st.session_state.get("or_key", "").strip())

                # статус и прогресс — прямо в карточке, без раскрытия expander'а
                if ss == "running":
                    st.progress(min(job.get("summary_progress") or 0.0, 1.0),
                                text="📝 " + (job.get("summary_stage") or "Генерация изложения…"))
                elif ss == "error" and job.get("summary_error"):
                    st.error(job["summary_error"][:3000])

                s1, _ = st.columns([3, 7])
                with s1:
                    label = "🔄 Обновить изложение" if sm.exists() else "📝 Составить изложение"
                    if st.button(label, key=f"sum-{jid}", disabled=ss == "running" or not can_sum):
                        lang = "ru" if st.session_state.get(
                            "sum_lang", "Русский (перевод)").startswith("Русский") else "orig"
                        m.start_summary(jid, st.session_state["or_key"].strip(),
                                        st.session_state.get("or_model", "openrouter/free"),
                                        lang=lang)
                        st.rerun(scope="fragment")
                if not can_sum:
                    st.caption("Чтобы составить изложение — укажите API-ключ OpenRouter в боковой панели")

                if sm.exists():
                    with st.expander("📝 Изложение", expanded=True):
                        st.markdown(sm.read_text(encoding="utf-8"))
                        st.download_button("⬇️ Итог (MD)", sm.read_bytes(),
                                           file_name=f"{jid}-summary.md", key=f"dl-{jid}-sum")


jobs_fragment()