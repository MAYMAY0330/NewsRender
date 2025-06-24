import json
import os
import asyncio
import re
from typing import Any, Dict, List

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_FILE = "data/recent_articles.json"
OUTPUT_FILE = "data/classified_articles_openai.json"
MODEL_NAME = "gpt-3.5-turbo"
MAX_CONTENT_TOKENS = 1000

PROMPT_TEMPLATE = """
You are an AI-powered news filter working at TPIsoftware, a software company based in Taiwan that specializes in AI development, enterprise platforms, and financial technologies.

Each day, your job is to help the product and strategy teams scan hundreds of global and East Asian news articles and identify only the ones that are relevant to our company’s interests.

We are specifically interested in news articles related to:

- Artificial Intelligence (AI), including new applications, tools, models, or platforms.
- Financial Technology (FinTech), such as digital banking, fraud detection, robo-advisors, AI in risk control or personal finance.
- Blockchain and Crypto technologies, especially their use in automation, intelligent agents, or AI-enhanced Web3 projects.

We **do not want** articles that are mainly about politics, lifestyle, sports, culture, general economy, or unrelated industries.

Please focus on whether the article has any real or potential connection to **AI, FinTech, or Blockchain innovation**, especially if it has product, funding, technical, or strategic value.

Respond with a single JSON object like this:
{ "keep": true }
or
{ "keep": false }
"""


def truncate_text(text: str, max_tokens: int = MAX_CONTENT_TOKENS) -> str:
    words = text.split()
    return " ".join(words[:max_tokens])


def load_articles(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON in {path}")


def _parse_response(text: str) -> bool:
    try:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        else:
            text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return data.get("keep", False)
    except Exception as exc:
        print("⚠️ Failed to parse model response:", exc)
        print("🧪 Raw text was:", repr(text))
        return False


semaphore = asyncio.Semaphore(3)


async def check_relevance(article: Dict[str, Any]) -> bool | None:
    title = article.get("title", "")
    content = article.get("content") or article.get("description", "")
    if not title or not content:
        return None
    short_content = truncate_text(content)
    prompt = f"{PROMPT_TEMPLATE.strip()}\n\nTitle: {title}\n\nArticle Content:\n{short_content}"
    messages = [
        {
            "role": "system",
            "content": "You are a JSON-only API that determines if an article is relevant to AI, FinTech, or Blockchain.",
        },
        {"role": "user", "content": prompt},
    ]
    async with semaphore:
        try:
            resp = await openai.ChatCompletion.acreate(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
            )
            text = resp.choices[0].message.content
            print("📩 Model raw response:", text)
            return _parse_response(text)
        except Exception as exc:
            print(f"❌ OpenAI request failed: {exc}")
            return None


async def main_async() -> None:
    articles = load_articles(INPUT_FILE)
    valid_articles = [a for a in articles if a.get("title") and (a.get("content") or a.get("description"))]
    results: List[Dict[str, Any]] = []

    tasks = [check_relevance(a) for a in valid_articles]
    responses = await asyncio.gather(*tasks)

    for art, keep in zip(valid_articles, responses):
        if keep:
            results.append(art)
        elif keep is None:
            print(f"⚠️ Skipped article due to LLM error: {art['title']}")

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote {len(results)} relevant articles to {OUTPUT_FILE}")
    print(f"🧠 OpenAI 判定為相關的文章數量: {len(results)}")


if __name__ == "__main__":
    asyncio.run(main_async())
