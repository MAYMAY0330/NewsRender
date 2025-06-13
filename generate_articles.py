import json
import requests

NEWSAPI_KEY = "9d217822f10348cda2442c455e120ef2"
GNEWS_KEY = "0e5e3a8d6f331bfdc614553393804bae"

NEWSAPI_URL = "https://newsapi.org/v2/everything"
GNEWS_URL = "https://gnews.io/api/v4/search"


def fetch_newsapi():
    params = {
        "q": "AI Fintech",
        "language": "en",
        "pageSize": 10,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
    }
    response = requests.get(NEWSAPI_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    articles = []
    for a in data.get("articles", []):
        articles.append(
            {
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
                "source": {"name": a.get("source", {}).get("name")},
                "publishedAt": a.get("publishedAt"),
            }
        )
    return articles


def fetch_gnews():
    params = {
        "q": "AI OR Fintech OR Crypto",
        "lang": "en",
        "country": "us",
        "max": 10,
        "apikey": GNEWS_KEY,
    }
    response = requests.get(GNEWS_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    articles = []
    for a in data.get("articles", []):
        articles.append(
            {
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
                "source": {"name": a.get("source", {}).get("name")},
                "publishedAt": a.get("publishedAt"),
            }
        )
    return articles


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    newsapi_articles = fetch_newsapi()
    save_json("data/newsapi_articles.json", newsapi_articles)

    gnews_articles = fetch_gnews()
    save_json("data/gnews_articles.json", gnews_articles)


if __name__ == "__main__":
    main()
