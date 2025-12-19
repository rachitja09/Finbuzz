import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from requests.exceptions import RequestException, SSLError


def fetch_news_with_sentiment(api_key: str, tickers: list[str]) -> pd.DataFrame:
    query = " OR ".join(tickers) if tickers else "stock market"
    try:
        resp = requests.get("https://newsapi.org/v2/everything", params={
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 5,
            "apiKey": api_key
        }, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
    except (SSLError, RequestException):
        articles = []
    except Exception:
        articles = []

    analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None

    rows = []
    for art in articles:
        try:
            title = art.get("title", "")
            if not title:
                continue
            if analyzer is not None:
                score = analyzer.polarity_scores(title)["compound"]
            else:
                # fallback simple heuristic
                txt = title.lower()
                score = float(1 if "good" in txt or "positive" in txt else -1 if "bad" in txt or "negative" in txt else 0)
            label = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
            rows.append({"headline": title, "sentiment": score, "label": label})
        except Exception:
            continue
    return pd.DataFrame(rows)
