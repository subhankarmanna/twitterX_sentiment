# step2_vectorize.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib, os, re

BASE_PATH = os.path.join(os.path.dirname(__file__), "models")

# ✅ Find next version folder (v1, v2, v3…)
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)
    next_version = "v1"
else:
    existing = [d for d in os.listdir(BASE_PATH) if re.match(r"v\d+$", d)]
    if not existing:
        next_version = "v1"
    else:
        nums = [int(x[1:]) for x in existing]  # 'v3' -> 3
        next_version = f"v{max(nums) + 1}"

VERSION_PATH = os.path.join(BASE_PATH, next_version)
os.makedirs(VERSION_PATH, exist_ok=True)

print(f"🚀 Creating Vectorizer for {next_version}")

# ✅ Load cleaned dataset
df = pd.read_csv('Data/cleaned_sentiment_data.csv')
df = df.dropna(subset=['clean_text', 'sentiment'])

X = df['clean_text'].astype(str)
y = df['sentiment']

# ✅ Improved TF-IDF
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=3,
    max_df=0.90
)

X_vectorized = vectorizer.fit_transform(X)
print("✅ TF-IDF complete:", X_vectorized.shape)

# ✅ Save vectorizer + data for training script
joblib.dump(vectorizer, f"{VERSION_PATH}/tfidf_vectorizer.pkl")
joblib.dump(X_vectorized, f"{VERSION_PATH}/X_vectorized.pkl")
joblib.dump(y, f"{VERSION_PATH}/y_labels.pkl")

print(f"✅ Saved inside {VERSION_PATH}")
