# Train_Model.py

import joblib, os, re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

BASE_PATH = os.path.join(os.path.dirname(__file__), "models")

# ✅ Identify latest version folder (v1, v2, v3…)
version_dirs = [
    d for d in os.listdir(BASE_PATH)
    if os.path.isdir(os.path.join(BASE_PATH, d)) and re.match(r"v\d+$", d)
]

if not version_dirs:
    raise RuntimeError("❌ No version folders found in 'python/models'. Run step2_vectorize.py first.")

nums = [int(d[1:]) for d in version_dirs]
latest_version = f"v{max(nums)}"
VERSION_PATH = os.path.join(BASE_PATH, latest_version)

print(f"🚀 Training Sentiment Model for {latest_version}")

# ✅ Load vectorized data
X = joblib.load(f"{VERSION_PATH}/X_vectorized.pkl")
y = joblib.load(f"{VERSION_PATH}/y_labels.pkl")
print(f"✅ Data loaded — X: {X.shape}, y: {len(y)}")

# ✅ Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Train Logistic Regression
model = LogisticRegression(max_iter=400)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# ✅ Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"📊 Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ Save model
joblib.dump(model, f"{VERSION_PATH}/sentiment_model.pkl")
print(f"✅ Model saved at '{VERSION_PATH}/sentiment_model.pkl'")
