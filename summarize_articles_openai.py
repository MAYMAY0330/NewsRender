import json
import os
import math
from typing import Dict, List
import logging
import asyncio

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.ERROR)

INPUT_FILE = "data/news_data_openai.json"
OUTPUT_FILE = "data/news_data_openai.json"
MODEL_NAME = "gpt-3.5-turbo"


def load_articles() -> List[Dict]:
    if not os.path.exists(INPUT_FILE):
        return []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON in {INPUT_FILE}")


semaphore = asyncio.Semaphore(3)


def _parse_summary(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Summary:"):
            return line.replace("Summary:", "").strip()
    return text.strip()


async def openai_summarize(title: str, body: str) -> str:
    prompt = f"""
You are a bilingual AI assistant working for TPIsoftware, a Taiwan-based company specializing in enterprise platforms, AI development, and financial technologies.

Your task is to help internal teams — including product managers, developers, and data scientists — quickly understand news articles related to AI, FinTech, and emerging technology. Your summaries will appear in our internal daily news digest.

You will be given full-text news articles in either English or Chinese.

## Your Goals:
1. Summarize the article in **Traditional Chinese**, using up to **4 concise sentences**.
2. Be accurate — do **not invent** or infer information not in the original article.
3. Keep important **technical terms**, and include the **English term in parentheses** if needed.
4. Avoid generic phrases like "the company" or "the startup" — use specific names and dates when available.

## Input
**Title**:
{title}

**Article Content**:
{body}

## Output (in the following format):
---
Summary: [Your Traditional Chinese summary here]
"""
    messages = [
        {
            "role": "system",
            "content": "Return only a short summary in Traditional Chinese.",
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
            return _parse_summary(text)
        except Exception as exc:
            logging.error("OpenAI API call failed: %s", exc)
            return ""


async def main_async() -> None:
    articles = load_articles()
    summarized = []

    valid_articles = [a for a in articles if a.get('title') and a.get('content')]

    tasks = [openai_summarize(a['title'], a['content']) for a in valid_articles]
    summaries = await asyncio.gather(*tasks)

    for art, summary_zh in zip(valid_articles, summaries):
        if not summary_zh:
            continue

        region = art.get('region') or "Global"
        category = art.get('category') or "General Tech & Startups"

        print("✅ Title:", art['title'])
        print("📝 Category:", category)
        print("🈶 Summary:", summary_zh)
        print("-----")

        src = art.get('source')
        if isinstance(src, dict):
            src = src.get('name')
        source_name = src or 'Unknown Source'

        published = art.get('publishedAt') or art.get('published_at')

        word_count = len(art['content'].split())
        read_time_min = max(1, math.ceil(word_count / 200))

        summarized.append({
            'region': region,
            'category': category,
            'title': art['title'],
            'summary_zh': summary_zh,
            'source': source_name,
            'read_time': f"{read_time_min} min read",
            'url': art.get('url'),
            'published_at': published,
            'tags': []
        })

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(summarized, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote summaries to {OUTPUT_FILE}")
    print(f"📝 OpenAI 成功摘要的文章總數: {len(summarized)}")


if __name__ == '__main__':
    asyncio.run(main_async())
