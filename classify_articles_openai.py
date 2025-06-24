"""OpenAI-based article classifier.

This runs after ``filter_relevance_openai.py`` and assigns a category
and region to each article. The results are saved for later summarization.
"""

import json
import os
import re
import asyncio
from typing import Dict, List, Any

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_FILE = "data/classified_articles_openai.json"
OUTPUT_ALL_FILE = "data/news_data_openai.json"
CATEGORY_DIR = "data/categorized"
MODEL_NAME = "gpt-3.5-turbo"

CATEGORIES = [
    "General Tech & Startups",
    "Applied AI & FinTech",
    "Blockchain & Crypto",
]

REGIONS = ["Global", "East Asia"]
MAX_CONTENT_TOKENS = 1000


def truncate_text(text: str, max_tokens: int = MAX_CONTENT_TOKENS) -> str:
    words = text.split()
    return " ".join(words[:max_tokens])


PROMPT_TEMPLATE = """
You are a news classification AI assistant working for the AI Innovation Department at TPIsoftware, a Taiwan-based software company specializing in artificial intelligence, financial technology, and enterprise software solutions.

Our team is building an internal news intelligence system to help product managers, researchers, and engineers stay updated on real-world applications of AI and emerging technologies across global and East Asian markets.

Your task is to analyze each article and assign:
1. One **category** — the article’s primary topic
2. One **region** — where the article is geographically focused

---

🎯 Why this matters:
Your classifications help us:
- Identify real-world use cases of AI and FinTech
- Track innovation across Asia and the world
- Surface relevant news for internal strategy, product planning, and technical research

---

🧠 **Categories (choose one only):**

- **General Tech & Startups**
  For news about general technology trends, enterprise tools, consumer apps, or startup activity not directly focused on AI, finance, or crypto.
  *Example: A startup launches a productivity tool or a SaaS company raises funding.*

- **Applied AI & FinTech**
  For articles about practical uses of artificial intelligence or financial technology. Includes LLM applications, algorithmic trading, AI customer service, robo-advisors, fraud detection, etc.
  *Example: A bank uses a large language model to automate customer service.*

- **Blockchain & Crypto**
  For content about crypto exchanges, smart contracts, Web3 infrastructure, blockchain applications in finance, or CBDCs.
  *Example: A government announces a pilot program for a digital currency.*

---

🌍 **Regions (choose one only):**

- **East Asia**
  For news focused on Taiwan, China, Japan, South Korea, or Hong Kong.
  *This region is our strategic priority and we track its trends closely.*

- **Global**
  For all other regions (e.g. U.S., Europe, India), or if the article discusses global tech trends, multinational initiatives, or broad international applications.

---

📤 **Output Format:**

Return only a JSON object on a single line, with no formatting or explanation:
{"category": "Applied AI & FinTech", "region": "East Asia"}
"""


def load_articles(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON in {path}")


semaphore = asyncio.Semaphore(3)


def _parse_response(text: str) -> Dict[str, Any]:
    try:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        else:
            text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return {
            "category": data.get("category", ""),
            "region": data.get("region", "Global"),
        }
    except Exception as exc:
        print("⚠️ Failed to parse model response:", exc)
        print("🧪 Raw text was:", repr(text))
        return {"category": "", "region": "Global"}


async def classify_article(article: Dict[str, Any]) -> Dict[str, Any] | None:
    title = article.get("title", "")
    content = article.get("content") or article.get("description", "")
    if not title or not content:
        return None
    short_content = truncate_text(content)
    prompt = f"{PROMPT_TEMPLATE.strip()}\n\nTitle: {title}\n\nArticle Content:\n{short_content}"
    messages = [
        {
            "role": "system",
            "content": "You are a JSON-only API that assigns a category and region to the article.",
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
    os.makedirs(CATEGORY_DIR, exist_ok=True)

    grouped = {region: {cat: [] for cat in CATEGORIES} for region in REGIONS}
    results: List[Dict[str, Any]] = []

    valid_articles = [a for a in articles if a.get("title") and (a.get("content") or a.get("description"))]

    tasks = [classify_article(a) for a in valid_articles]
    responses = await asyncio.gather(*tasks)

    for art, result in zip(valid_articles, responses):
        if not result:
            continue
        art["category"] = result.get("category", "")
        art["region"] = result.get("region", "Global")
        results.append(art)
        cat = art["category"]
        region = art["region"]
        if region in grouped and cat in grouped[region]:
            grouped[region][cat].append(art)

    with open(OUTPUT_ALL_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    for region, cats in grouped.items():
        for cat, items in cats.items():
            safe_cat = cat.lower().replace(" ", "_").replace("&", "and")
            filename = f"{region.lower()}_{safe_cat}.json"
            path = os.path.join(CATEGORY_DIR, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
    print(
        f"Wrote {len(results)} articles with classifications to {OUTPUT_ALL_FILE}"
    )
    print(f"🏷️ OpenAI 成功分類的文章數量: {len(results)}")


if __name__ == "__main__":
    asyncio.run(main_async())
