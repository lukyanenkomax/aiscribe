"""Изложение транскрипта с помощью LLM через OpenRouter.

Для многочасовых записей используется map-reduce:
- длинный текст разбивается на фрагменты (~12 тыс. символов);
- каждый фрагмент сжимается отдельным запросом;
- из сжатых заметок собирается итоговое изложение.

Промпт явно учитывает, что текст получен распознаванием речи и содержит
орфографические/фонетические ошибки, пропущенную пунктуацию и т.п.
"""

import json
import time
import urllib.error
import urllib.request
from collections import Counter

API_URL = "https://openrouter.ai/api/v1/chat/completions"
CHUNK_CHARS = 12000  # ~3-4 тыс. токенов — запас под контекст бесплатных моделей
REDUCE_MAX_CHARS = 50000  # больше этого размера заметки сжимаются промежуточным шагом

BASE_PROMPT = """\
Ты работаешь с автоматической транскрибацией речи — текстом, полученным \
распознаванием речи, а не вычитанным человеком. В нём возможны ошибки:
— орфографические, опечатки, ошибки в роде/падеже, пропущенная пунктуация, \
отсутствие заглавных букв;
— искажения слов, похожих по звучанию (омофоны), перестановки букв;
— повторы, слова-паразиты («эээ», «ну», «как бы»), оборванные и повторённые фразы;
— пропущенные или лишние слова, разорванные на куски предложения.

Не цитируй эти ошибки дословно и не комментируй их — восстанавливай смысл по \
контексту, как это сделал бы внимательный редактор. При этом не додумывай \
факты, которых нет в тексте: опирайся только на его содержание.

Строго запрещено: повторять одну и ту же строку или фразу много раз, выходить \
на цикл, многократно перечислять одно утверждение. Каждая строка ответа должна \
быть содержательно новой; если чувствуешь, что повторяешься — остановись и \
переходи к следующему пункту."""


def _task(lang: str) -> str:
    """Инструкция о языке итогового изложения (вставляется в {task})."""
    if lang == "orig":
        return ("Пиши весь ответ строго на языке оригинала — языке этого текста. "
                "Если в формате есть русские заголовки — не копируй их, "
                "сделай аналогичные заголовки на языке оригинала.")
    return "Пиши изложение на русском языке (имена, термины и прямые цитаты оставляй в оригинале)."


INTER_PROMPT = BASE_PROMPT + """

Ниже — сжатые заметки по фрагментам записи. Сведи их в связные заметки \
(Markdown, до 700 слов), сохранив ВСЕ содержательные тезисы, факты, цифры, \
имена и хронологию. Ничего не отбрасывай и не сокращай число пунктов — \
объединять можно только буквальные повторы. {task}
Не повторяй одну и ту же строку; каждая строка — новое содержание.

Заметки:
"""

REDUCE_PROMPT = BASE_PROMPT + """

Ниже — сжатые заметки по фрагментам длинной записи (порядок сохранён).
Составь единое связное изложение в формате Markdown. {task}

Главное требование — ПОЛНОТА: изложение должно содержать все содержательные \
тезисы из заметок, а не только самые главные. Для длинной записи это обычно \
15–40 пунктов. Не отбрасывай детали, цифры, имена, названия и оценки; не \
объединяй разные тезисы в один пункт; не заменяй конкретику общими словами.

## Краткое содержание
5–8 предложений о том, о чём запись целиком.

## Основные тезисы
Подробный маркированный список ключевых мыслей и аргументов, с промежуточными \
подзаголовками по темам, если запись охватывает несколько тем.

## Ключевые факты, цифры и имена
Только то, что действительно названо в тексте — перечисли как можно полнее.

## Выводы
Выводы, рекомендации или итоги — если они есть; иначе раздел можно опустить.

Только важные цитаты вставляй дословно, исправляя в них орфографию.

Заметки по фрагментам:
"""

FULL_PROMPT = BASE_PROMPT + """

Ниже — полный текст автоматической транскрибации записи.
Составь единое связное изложение в формате Markdown. {task}

Главное требование — ПОЛНОТА: изложение должно содержать все содержательные \
тезисы текста, а не только самые главные. Не отбрасывай детали, цифры, имена, \
названия и оценки; не заменяй конкретику общими словами.

## Краткое содержание
5–8 предложений о том, о чём запись целиком.

## Основные тезисы
Подробный маркированный список ключевых мыслей и аргументов, с промежуточными \
подзаголовками по темам, если запись охватывает несколько тем.

## Ключевые факты, цифры и имена
Только то, что действительно названо в тексте — перечисли как можно полнее.

## Выводы
Выводы, рекомендации или итоги — если они есть; иначе раздел можно опустить.

Только важные цитаты вставляй дословно, исправляя в них орфографию.

Текст транскрипции:
"""

PART_PROMPT = BASE_PROMPT + """

Это фрагмент длинной записи. Сделай подробный конспект в Markdown (до 700 слов): \
выдели ВСЕ содержательные тезисы, факты, цифры, имена и важные цитаты фрагмента, \
сохрани порядок и логику изложения. Ничего не отбрасывай и не сокращай до \
нескольких главных мыслей; если в фрагменте несколько сюжетов — оформи их \
подзаголовками.

Фрагмент:
"""


def _content_text(message) -> str:
    """Достать текст ответа: content может быть строкой, списком частей или None."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _is_degenerate(text: str) -> bool:
    """Похоже ли на вырожденный ответ: одна строка повторяется много раз."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 8:
        return False
    top, n = Counter(lines).most_common(1)[0]
    return n >= 8 and n >= len(lines) * 0.12


def _chat(messages: list[dict], api_key: str, model: str,
          timeout: int = 300, retries: int = 2, max_tokens: int = 4000) -> str:
    last_reason = ""
    for attempt in range(retries + 1):
        # при повторах поднимаем temperature — вырожденные повторы обычно
        # возникают на низкой температуре и не повторяются при новой попытке
        temperature = (0.3, 0.7, 0.9)[min(attempt, 2)]
        data = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(API_URL, data=data, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "Local Transcriber",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:1000]
            try:
                body = json.loads(body)["error"]["message"]
            except Exception:
                pass
            if e.code == 400 and "max_tokens" in body.lower() and max_tokens > 4000:
                # модель не поддерживает такой лимит вывода — снижаем
                max_tokens = 4000
                last_reason = f"HTTP 400: {body}"
                continue
            if e.code in (429, 500, 502, 503) and attempt < retries:
                last_reason = f"HTTP {e.code}: {body}"
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}") from None
        if payload.get("error"):
            raise RuntimeError(f"OpenRouter: {payload['error'].get('message', payload['error'])}")
        try:
            choice = payload["choices"][0]
            text = _content_text(choice["message"])
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Неожиданный ответ API: {payload}") from e
        truncated = choice.get("finish_reason") == "length"
        if not text.strip():
            # бесплатные модели иногда возвращают content: null — повторяем
            last_reason = str(payload.get("error") or "пустой content")
        elif _is_degenerate(text):
            last_reason = "модель зациклилась на повторении одной строки"
        elif truncated and max_tokens < 8000 and attempt < retries:
            # ответ обрезан по лимиту длины — просим больше места
            max_tokens = 8000
            last_reason = "ответ обрезан лимитом длины"
            continue
        else:
            return text
        if attempt < retries:
            time.sleep(2)
    raise RuntimeError(
        f"OpenRouter: модель «{model}» — {last_reason} (попытки исчерпаны). "
        "Попробуйте ещё раз или выберите другую модель.")


def _split(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Разбить текст на фрагменты по строкам (субтитры/абзацы), не рвя строки."""
    parts, cur = [], ""
    for line in text.splitlines():
        if cur and len(cur) + len(line) + 1 > limit:
            parts.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur)
    return parts


def summarize(text: str, api_key: str, model: str,
              progress_cb=None, lang: str = "ru") -> str:
    """Изложение транскрипта. progress_cb(frac, msg) — для отображения в UI.

    lang="ru" — на русском, lang="orig" — на языке оригинала записи.
    """
    text = text.strip()
    if not text:
        raise RuntimeError("Транскрипт пуст")

    task = _task(lang)

    def prog(frac, msg):
        if progress_cb:
            progress_cb(frac, msg)

    parts = _split(text)
    if len(parts) == 1:
        prog(0.3, "Запрос к LLM…")
        out = _chat([{"role": "user", "content": FULL_PROMPT.replace("{task}", task) + text}],
                    api_key, model, max_tokens=6000)
        prog(1.0, "Готово")
        return out

    notes = []
    for i, part in enumerate(parts, 1):
        prog((i - 1) / (len(parts) + 1), f"Сжатие фрагмента {i}/{len(parts)}")
        note = _chat([{"role": "user", "content": PART_PROMPT + part}],
                     api_key, model)
        notes.append(note)

    # заметки из многих фрагментов ужимаем пачками — иначе итоговый запрос
    # становится гигантским и бесплатные модели на нём зацикливаются
    batches, cur, cur_len = [], [], 0
    for note in (n for n in notes if n):
        if cur and cur_len + len(note) > REDUCE_MAX_CHARS:
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append(note)
        cur_len += len(note) + 6
    if cur:
        batches.append(cur)
    if len(batches) > 1:
        notes = []
        for i, batch in enumerate(batches, 1):
            prog(len(parts) / (len(parts) + 1) * i / len(batches),
                 f"Сжатие заметок {i}/{len(batches)}")
            notes.append(_chat(
                [{"role": "user",
                  "content": INTER_PROMPT.replace("{task}", task)
                  + "\n\n---\n\n".join(batch)}],
                api_key, model))
    joined = "\n\n---\n\n".join(n for n in notes if n)
    prog(len(parts) / (len(parts) + 1), "Сборка итогового изложения…")
    out = _chat([{"role": "user", "content": REDUCE_PROMPT.replace("{task}", task) + joined}],
                api_key, model, max_tokens=6000)
    prog(1.0, "Готово")
    return out
