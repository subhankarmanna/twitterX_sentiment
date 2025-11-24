# Twitter/X Sentiment Analysis  
<div align="center">  
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg" alt="Twitter Blue Logo" width="100" height="100">  
  <h3>⚡ High-Performance Sentiment Extraction Engine ⚡</h3>  
  <p><i>Streamlined · Reproducible · Dependency-Free</i></p>  
</div>  

---

## 🚀 Introduction  
**Twitter/X Sentiment Analysis** is a modular framework designed to deliver **fast and reliable sentiment insights** from tweet datasets.  
Built for **clarity and performance**, this project eliminates heavy Kaggle dependencies in favor of a **pure, streamlined Python architecture**.  

Whether for **academic research**, **competitive analysis**, or **industrial showcasing**, this tool converts raw text into **actionable data signals**.  

---

## ✨ Key Features  
- 🎯 **Tri-Class Classification** → Predicts **Positive**, **Negative**, and **Neutral** sentiments.  
- ⚙️ **Modular Architecture** → Extensible codebase for easy scaling and customization.  
- 🛡️ **Privacy-First Design** → Works with custom CSV datasets (tweet text only).  
- 🔓 **Open Source Ecosystem** → Licensed under **GNU GPL v3.0**.  

---

## 📂 System Architecture  
```plaintext
python/
├── Data/
│   ├── cleaned_sentiment_data.csv        # Preprocessed dataset
│   └── training.1600000.processed.csv    # High-volume corpus (No Emoticons)
│
├── models/
│   └── v1/                               # Serialized ML models (.pkl)
│
├── analyzer.py                           # Inference engine (Real-time prediction)
├── Data_set_Scrap.py                     # ETL: Scraping & Cleaning utilities
├── date_vectorize.py                     # NLP: TF-IDF Vectorization pipelines
├── pkl_viewer.py                         # Diagnostics: Model evaluation & inspection
└── Train_Model.py                        # Training loop & Model persistence
```

---

## 🛠️ Tech Stack  
| Component        | Technology     | Description |
|------------------|---------------|-------------|
| **Core**         | Python 3.x     | Runtime environment |
| **Data**         | pandas         | High-performance data structures |
| **ML**           | scikit-learn   | Algorithms for classification & regression |
| **NLP**          | TfidfVectorizer| Text vectorization |
| **Persistence**  | joblib         | Efficient model serialization |

---

## ⚡ Getting Started  

### 1️⃣ Install Dependencies  
```bash
pip install pandas scikit-learn joblib
```

### 2️⃣ Workflow Pipeline  
**Step 1: Data Ingestion**  
```bash
python Data_set_Scrap.py
```

**Step 2: Feature Extraction**  
```bash
python date_vectorize.py
```

**Step 3: Model Training**  
```bash
python Train_Model.py
```

**Step 4: Inference & Diagnostics**  
```bash
# Real-time sentiment labeling
python analyzer.py  

# Model inspection
python pkl_viewer.py
```

---

## 📜 License & Data Policy  
- Licensed under **GNU GPL v3.0** → Free to use, modify, and distribute.  
- Uses **pre-cleaned datasets** → No raw Twitter API data, compliant with X/Twitter’s developer policy.  
- Logo image licensed under **Creative Commons Attribution-Share Alike 4.0 International**.  

---

## 🤝 Contributing  
Open-source spirit drives innovation! Contributions are welcome:  

1. Fork the Project  
2. Create your Feature Branch → `git checkout -b feature/AmazingFeature`  
3. Commit Changes → `git commit -m 'Add some AmazingFeature'`  
4. Push to Branch → `git push origin feature/AmazingFeature`  
5. Open a Pull Request  

---

## 📬 Contact  
For **enterprise inquiries**, **research collaboration**, or **support**:  
- Open a GitHub Issue  
- Contact the maintainer directly  
