import os
import joblib
import re

MODELS_BASE = os.path.join(os.path.dirname(__file__), "models")

# ✅ Find latest version: v1, v2, v3...
version_dirs = [
    d for d in os.listdir(MODELS_BASE)
    if os.path.isdir(os.path.join(MODELS_BASE, d)) and re.match(r"v\d+$", d)
]

if not version_dirs:
    raise RuntimeError("❌ No version folders found in 'models/'. Please run step2_vectorize.py + Train_Model.py first.")

version_nums = [int(d[1:]) for d in version_dirs]   # 'v3' -> 3
latest_version = f"v{max(version_nums)}"
VERSION_DIR = os.path.join(MODELS_BASE, latest_version)

MODEL_PATH = os.path.join(VERSION_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(VERSION_DIR, "tfidf_vectorizer.pkl")

print(f"🔄 Loading model and vectorizer from {latest_version} ...")

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

print(f"✅ Loaded sentiment model & TF-IDF vectorizer from {latest_version}.\n")


# ---------- Text cleaning ----------
import re as _re

def basic_clean(text: str) -> str:
    text = text.lower()
    text = _re.sub(r"http\S+", " ", text)
    text = _re.sub(r"@\w+", " ", text)
    text = _re.sub(r"[^a-z0-9\s]", " ", text)
    text = _re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(text: str):
    cleaned = basic_clean(text)
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    label_id = int(proba.argmax())
    label_text = "Positive" if label_id == 1 else "Negative"
    confidence = float(proba.max())
    return label_id, label_text, confidence


def predict_batch(text_list):
    cleaned_list = [basic_clean(t) for t in text_list]
    X = vectorizer.transform(cleaned_list)
    probas = model.predict_proba(X)
    labels = probas.argmax(axis=1)

    results = []
    for original, label_id, proba in zip(text_list, labels, probas):
        label_text = "Positive" if label_id == 1 else "Negative"
        results.append({
            "text": original,
            "label_id": int(label_id),
            "label": label_text,
            "confidence": float(proba.max())
        })
    return results


if __name__ == "__main__":
    print(f"🧪 Twitter Sentiment Analyzer ({latest_version})")
    print("Type 'q' to quit.\n")

    while True:
        tweet = input("Enter a tweet: ")

        if tweet.lower().strip() in ("q", "quit", "exit"):
            print("👋 Exiting analyzer.")
            break

        label_id, label_text, conf = predict_sentiment(tweet)
        print(f"👉 Sentiment: {label_text} (label={label_id}, confidence={conf:.3f})\n")
