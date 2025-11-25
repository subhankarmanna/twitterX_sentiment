# clean_data.py  (replace your existing script contents with this)
import os
import pandas as pd
import re

# --- Use project-relative absolute paths so we never create python/python etc ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
INPUT_CSV = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "cleaned_sentiment_data.csv")

# Step 1: Load CSV file (explicit encoding)
df = pd.read_csv(INPUT_CSV, encoding='latin-1', header=None)
print("✅ Loaded:", INPUT_CSV)

# Step 2: Assign proper column names
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Step 3: Keep only the columns we need
df = df[['sentiment', 'text']]

# Step 4: Convert sentiment values: 0 = negative, 4 = positive
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})

# Step 5: Clean the text using regex (improved)
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # remove URLs and mentions
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    # keep letters and spaces (you can relax this to keep emojis if needed)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Step 6: Apply cleaning to the text column
df['clean_text'] = df['text'].apply(clean_text)

# Optional: drop rows with empty clean_text
df = df[df['clean_text'].str.strip() != ""]

# Save cleaned file
df[['sentiment', 'clean_text']].to_csv(OUTPUT_CSV, index=False)
print("✅ Cleaned saved to:", OUTPUT_CSV)

# Step 7: Show a few samples
print(df[['text', 'clean_text']].head())
