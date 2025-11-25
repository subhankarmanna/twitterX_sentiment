import os
import joblib
import re

# ---------- Auto-detect latest version folder (v1, v2, v3...) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # python/
MODELS_BASE = os.path.join(BASE_DIR, "models")

version_dirs = [
    d for d in os.listdir(MODELS_BASE)
    if os.path.isdir(os.path.join(MODELS_BASE, d)) and re.match(r"v\d+$", d)
]

if not version_dirs:
    raise RuntimeError("❌ No version folders found in 'models/'")

version_nums = [int(d[1:]) for d in version_dirs]
latest_version = f"v{max(version_nums)}"
VERSION_DIR = os.path.join(MODELS_BASE, latest_version)

MODEL_PATH = os.path.join(VERSION_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(VERSION_DIR, "tfidf_vectorizer.pkl")

print(f"🔄 Loading model and vectorizer from {latest_version} ...")
vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)
print(f"✅ Loaded TF-IDF & Model ({latest_version}).\n")


# ---------- Lexicons ----------
GREETINGS = {"hello", "hi", "hey", "greetings", "gm", "gn", "good", "morning", "evening", "goodnight"}

NEG_WORDS = {
    "depression","depressed","suicidal","hate","angry","sad","anxious","anxiety",
    "sucks","terrible","lonely","worst","bad","awful","broken"
}

POS_WORDS = {
    "amazing","great","love","awesome","best","good","excellent","fantastic",
    "happy","nice","perfect","like","loved","beautiful","smooth"
}

NEGATION_WORDS = {"not","n't","never","no","dont","don't"}

CONFIDENCE_NEUTRAL_THRESHOLD = 0.60


# ---------- Normal Text Cleaning ----------
def basic_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove leading greetings
    text = re.sub(r'^(?:hello|hi|hey|greetings|gm|gn|good|morning|evening|goodnight)\b\s*', '', text)

    return text


# ---------- Negation Handling (NOT_good, NOT_bad) ----------
def apply_negation_marks(text: str) -> str:
    tokens = text.split()
    out = []
    negate = False
    window = 0

    for t in tokens:
        if t in NEGATION_WORDS:
            negate = True
            window = 2              # negate next 2 tokens
            out.append(t)
            continue

        if negate and window > 0:
            out.append("NOT_" + t)
            window -= 1
            if window == 0:
                negate = False
            continue

        out.append(t)

    return " ".join(out)


# ---------- Main Prediction ----------
def predict_sentiment(text: str):
    original = "" if text is None else str(text).strip()
    if original == "":
        return -1, "Neutral", 0.0

    # single greeting → neutral
    orig_tokens = original.lower().split()
    if len(orig_tokens) == 1 and orig_tokens[0] in GREETINGS:
        return -1, "Neutral", 0.50

    cleaned = basic_clean(text)
    if cleaned.strip() == "":
        return -1, "Neutral", 0.50

    # apply negations
    cleaned = apply_negation_marks(cleaned)

    # softening / hedging words that reduce negative polarity
    HEDGES = {"okay", "ok", "fine", "alright", "i guess", "maybe", "perhaps", "could", "bit", "a bit", "kinda", "sorta"}

    # tokens and lexicon counts (handle NOT_ markers from negation)
    tokens = cleaned.split()

    # count hedges (allow multi-word phrases by checking substring in cleaned)
    hedge_count = sum(1 for h in HEDGES if h in cleaned)

    pos_count = sum(1 for w in tokens if (w in POS_WORDS) or (w.startswith("NOT_") and w[4:] in NEG_WORDS))
    neg_count = sum(1 for w in tokens if (w in NEG_WORDS) or (w.startswith("NOT_") and w[4:] in POS_WORDS))

    # reduce negative strength by hedges (each hedge reduces one neg token effect)
    if hedge_count > 0 and neg_count > 0:
        neg_count = max(0, neg_count - hedge_count)

    # If hedges and no strong lexicon left -> treat as Neutral early
    if hedge_count > 0 and pos_count == 0 and neg_count == 0:
        return -1, "Neutral", 0.55

    # Mixed Case: both pos & neg present → use weighted logic (same as before but with adjusted counts)
    if pos_count > 0 and neg_count > 0:
        Xtemp = vectorizer.transform([cleaned])
        proba = model.predict_proba(Xtemp)[0]
        model_label = int(proba.argmax())
        model_conf = float(proba.max())

        # If hedges present and model not super-confident -> prefer Neutral
        if hedge_count > 0 and model_conf < 0.90:
            return -1, "Neutral", model_conf

        # low model confidence -> Neutral
        if model_conf < 0.60:
            return -1, "Neutral", model_conf

        # pos dominates
        if pos_count > neg_count:
            if model_label == 1 or model_conf < 0.95:
                return 1, "Positive", max(model_conf, 0.85)
            else:
                return 0, "Negative", model_conf

        # neg dominates
        if neg_count > pos_count:
            if model_label == 0 or model_conf < 0.95:
                return 0, "Negative", max(model_conf, 0.85)
            else:
                return 1, "Positive", model_conf

        # tie -> neutral safety
        if model_conf < 0.65:
            return -1, "Neutral", model_conf
        return model_label, ("Positive" if model_label == 1 else "Negative"), model_conf

    # Only Negative lexicon present (after hedge reduction)
    if neg_count > 0 and pos_count == 0:
        Xtemp = vectorizer.transform([cleaned])
        proba = model.predict_proba(Xtemp)[0]
        model_label = int(proba.argmax())
        model_conf = float(proba.max())
        # if hedges present, be conservative
        if hedge_count > 0 and model_conf < 0.90:
            return -1, "Neutral", model_conf
        if model_label == 1 and model_conf < 0.95:
            return 0, "Negative", max(model_conf, 0.90)
        return 0, "Negative", model_conf

    # Only Positive lexicon present
    if pos_count > 0 and neg_count == 0:
        Xtemp = vectorizer.transform([cleaned])
        proba = model.predict_proba(Xtemp)[0]
        model_label = int(proba.argmax())
        model_conf = float(proba.max())
        if model_label == 0 and model_conf < 0.95:
            return 1, "Positive", max(model_conf, 0.85)
        return 1, "Positive", model_conf

    # Only Negative words
    if neg_count > 0 and pos_count == 0:
        Xtemp = vectorizer.transform([cleaned])
        proba = model.predict_proba(Xtemp)[0]
        model_label = int(proba.argmax())
        model_conf = float(proba.max())
        if model_label == 1 and model_conf < 0.95:
            return 0, "Negative", max(model_conf, 0.90)
        return 0, "Negative", model_conf

    # Only Positive words
    if pos_count > 0 and neg_count == 0:
        Xtemp = vectorizer.transform([cleaned])
        proba = model.predict_proba(Xtemp)[0]
        model_label = int(proba.argmax())
        model_conf = float(proba.max())
        if model_label == 0 and model_conf < 0.95:
            return 1, "Positive", max(model_conf, 0.85)
        return 1, "Positive", model_conf


    # Model-only fallback
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    model_label = int(proba.argmax())
    model_conf = float(proba.max())

    if model_conf < CONFIDENCE_NEUTRAL_THRESHOLD:
        return -1, "Neutral", model_conf

    return model_label, ("Positive" if model_label == 1 else "Negative"), model_conf


# --------- Batch prediction ----------
def predict_batch(text_list):
    results = []
    for t in text_list:
        label_id, label_text, confidence = predict_sentiment(t)
        results.append({
            "text": t,
            "label_id": int(label_id),
            "label": label_text,
            "confidence": float(confidence)
        })
    return results


# ---------- CLI ----------
if __name__ == "__main__":
    print(f"🧪 Twitter Sentiment Analyzer ({latest_version})\n")
    print("Type 'q' to quit.\n")

    while True:
        tweet = input("Enter a tweet: ")
        if tweet.lower().strip() in ("q", "quit", "exit"):
            print("👋 Exiting analyzer.")
            break

        label_id, label_text, conf = predict_sentiment(tweet)
        print(f"👉 Sentiment: {label_text} (label={label_id}, confidence={conf:.3f})\n")
