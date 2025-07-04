import os
import sys
import subprocess
from datetime import datetime

# Define all steps and expected output files
STEPS = [
    
    ("fetch_rss_articles.py", "data/rss_articles.json"),
    ("filter_articles_by_date.py", "data/recent_articles.json"),
    ("filter_relevance_gpt.py", "data/classified_articles.json"),
    ("classify_articles_gpt.py", "data/news_data.json"),
    ("select_top_articles.py", "data/selected_articles.json"),
    ("summarize_articles.py", "data/news_data.json"),
    ("validate_news_data.py", "data/news_data.json"),
    ("generate_digest.py", "digest.html"),
    ("send_digest.py", None),
]


def check_output_file(path: str) -> None:
    if not path:
        return
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"⚠️ Warning: {path} is missing or empty")


def run_step(index: int, script: str, output: str | None) -> None:
    print(f"🔧 [Step {index}] Running {script}...")
    try:
        result = subprocess.run(
            [sys.executable, script],
            check=True,
            env=os.environ.copy()  # ✅ pass all current environment variables
        )
        if result.returncode == 0:
            print(f"✅ [Step {index}] {script} completed")
        else:
            print(f"❌ [Step {index}] {script} exited with code {result.returncode}")
    except subprocess.CalledProcessError as exc:
        print(f"❌ [Step {index}] {script} failed: {exc}")
    except Exception as exc:
        print(f"❌ [Step {index}] {script} error: {exc}")
    finally:
        check_output_file(output)


def main() -> None:
    start = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"⏰ Starting Polaris Digest Run: {start}")
    for i, (script, output) in enumerate(STEPS, start=1):
        run_step(i, script, output)
    end = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"⏰ Polaris Digest Run finished: {end}")


if __name__ == "__main__":
    main()
