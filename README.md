<div align="center">

<img src="https://img.shields.io/badge/📡-Sentiment%20Radar-7c3aed?style=for-the-badge&labelColor=0a0a0f" alt="Sentiment Radar"/>

# Sentiment Radar

**On-demand social media sentiment analysis powered by a custom BiLSTM + Self-Attention deep learning model**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.1-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Keras](https://img.shields.io/badge/tf--keras-2.17.0-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=flat-square)]()
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-76.13%25-7c3aed?style=flat-square)]()
[![Dataset](https://img.shields.io/badge/Training%20Data-2.86M%20Comments-38bdf8?style=flat-square)]()

[**📺 Watch Demo**](https://youtu.be/ZBAwq79opZk) · [**📊 Model Details**](#-model-architecture) · [**⚡ Quick Start**](#-quick-start)

<br/>

</div>

---

## 📌 Overview


### What it does

- 🔍 **Scrapes on demand** — enter any topic, fetch up to 50 comments per platform in seconds
- 🧠 **Classifies with a custom model** — BiLSTM + Self-Attention trained on 2.86M comments from 6 datasets
- 📊 **Visualises everything** — donut charts, platform radar, sentiment gap bars, confidence histograms
- 💬 **Shows raw comments** — filterable by platform and sentiment with confidence scores per prediction
- 📥 **Exports results** — download full analysis as CSV

### Platforms supported

| Platform | Auth required | Status |
|---|---|---|
| Reddit | Optional (OAuth improves freshness) | ✅ Active |
| HackerNews | None | ✅ Active |
| Dev.to | None | ✅ Active |
| YouTube | Free API key | ✅ Optional |
| Mastodon | None | ✅ Optional |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SENTIMENT RADAR                             │
├──────────────────────┬──────────────────────┬───────────────────────┤
│     DATA LAYER       │    INFERENCE LAYER   │   PRESENTATION LAYER  │
│                      │                      │                       │
│  scraper.py          │  inference.py        │  streamlit_app.py     │
│  ┌──────────────┐    │  ┌───────────────┐   │  ┌─────────────────┐  │
│  │   Reddit     │    │  │ Text Cleaning │   │  │  KPI Cards      │  │
│  │   HackerNews │───▶│  │ Negation Tag  │───▶│  │  Donut Charts   │  │
│  │   Dev.to     │    │  │ Tokenize+Pad  │   │  │  Radar Chart    │  │
│  │   YouTube    │    │  │ BiLSTM Model  │   │  │  Gap Analysis   │  │
│  │   Mastodon   │    │  │ Softmax(3)    │   │  │  Comment Cards  │  │
│  └──────────────┘    │  └───────────────┘   │  └─────────────────┘  │
└──────────────────────┴──────────────────────┴───────────────────────┘
```

---

## 🧠 Model Architecture

The core model is a **Bidirectional LSTM with Self-Attention and dual pooling**, trained from scratch on **2.86M social media comments** aggregated from 6 datasets.

```
Input (seq_len=100)
        │
        ▼
Embedding Layer (vocab=20,000 · dim=128)  →  2,560,000 params
        │
SpatialDropout1D(0.3)
        │
        ▼
BiLSTM(128 units, return_sequences=True)  →  263,168 params
        │
BiLSTM(64 units, return_sequences=True)   →  164,352 params
        │
        ▼
Self-Attention Layer
        │
   ┌────┴────┐
   ▼         ▼
GlobalAvg  GlobalMax
  Pool       Pool
   └────┬────┘
        │ Concat
        ▼
  Dense(128, ReLU, L2) → Dropout(0.5)
  Dense(64,  ReLU)     → Dropout(0.3)
        │
        ▼
  Dense(3, Softmax)
  [negative · neutral · positive]

  Total parameters: ~3,000,000
```

### 📊 Model Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **76.13%** |
| Test Loss | 0.5634 |
| Training Time | ~8,244 seconds |
| Epochs | 12 |

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negative | 0.77 | 0.79 | 0.78 | 239,513 |
| Neutral | 0.63 | 0.62 | 0.62 | 83,142 |
| Positive | 0.80 | 0.78 | 0.79 | 249,108 |
| **Weighted Avg** | **0.76** | **0.76** | **0.76** | 571,763 |

> The neutral class underperforms (F1: 0.62) due to class imbalance — neutral samples (~410K) are significantly fewer than positive/negative (~1.2M+ each). This is a known challenge in social media sentiment datasets.

### 📦 Training Data

| Dataset | Source | Size |
|---|---|---|
| Sentiment140 | Twitter | ~1.6M |
| Reddit Comments | Reddit | ~400K |
| IMDB Reviews | IMDB | ~50K |
| Amazon Reviews | Amazon | ~400K |
| YouTube Comments | YouTube | ~250K |
| HackerNews | HackerNews | ~160K |
| **Total** | **6 sources** | **~2.86M** |

**Class distribution:** Positive ~1.25M · Negative ~1.2M · Neutral ~0.41M

### 🔤 Preprocessing Pipeline

A key feature is the **negation scope tagging** — applied before tokenisation so the model learns `good` and `good_NEG` as distinct signals:

```
Input:  "I didn't enjoy this at all."
  │
  ├─ 1. Expand contractions  →  "I did not enjoy this at all."
  ├─ 2. Negation scope tag   →  "I did not enjoy_NEG this_NEG at_NEG all_NEG."
  ├─ 3. Strip non-alpha      →  "I did not enjoy_NEG this_NEG at_NEG all_NEG"
  └─ 4. Remove stopwords     →  "not enjoy_NEG all_NEG"
          (keeping: not, but, very, can, won't, never, etc.)

Output: "not enjoy_NEG all_NEG"   →  Predicted: NEGATIVE (98.93%)
```

**Confidence examples from batch testing:**

| Input | Prediction | Confidence |
|---|---|---|
| "This product is absolutely amazing!" | Positive | 91.22% |
| "Worst experience I've ever had." | Negative | 98.57% |
| "I never felt so happy in my life!" | Positive | 86.98% |
| "I don't like this at all." | Negative | 87.22% |

---

## 📁 Project Structure

```
Sentiment_Radar/
│
├── streamlit_app.py              ← Main dashboard entry point
│
├── app/
│   ├── __init__.py
│   ├── inference.py              ← Model loading, preprocessing, batched prediction
│   └── scraper.py                ← Multi-platform comment scraper
│
├── model_files/                  ← Place your 3 model artifacts here
│   ├── best_sentiment_model.keras   (~35 MB)
│   ├── tokenizerB.joblib            (~32 MB)
│   └── encoderB.joblib              (~0.5 MB)
│
├── nltk_data/                    ← Pre-downloaded NLTK corpora (committed)
│
├── .streamlit/
│   ├── config.toml               ← Dark theme configuration
│   └── secrets.toml              ← API keys (gitignored — never committed)
│
├── requirements.txt              ← Linux / Streamlit Cloud dependencies
├── requirements-windows.txt      ← Windows local dev dependencies
├── setup.sh                      ← Pre-start script for Streamlit Cloud
├── download_nltk.py              ← One-time NLTK data download
├── .env.example                  ← Environment variable template
├── .gitattributes                ← Line ending configuration
└── .gitignore
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10
- The 3 model files (`best_sentiment_model.keras`, `tokenizerB.joblib`, `encoderB.joblib`)

### 1. Clone the repository

```bash
git clone https://github.com/BhaveshN1015/Sentiment_Radar.git
cd Sentiment_Radar
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
# Windows
pip install -r requirements-windows.txt

# Linux / Mac
pip install -r requirements.txt
```

### 4. Add model files

Copy your 3 model artifacts into `model_files/`:

```
model_files/
├── best_sentiment_model.keras   (~35 MB)
├── tokenizerB.joblib            (~32 MB)
└── encoderB.joblib              (~0.5 MB)
```

### 5. Configure environment (optional)

```bash
copy .env.example .env      # Windows
cp .env.example .env        # Mac / Linux
```

Edit `.env` and add your API keys (all optional — app works without them):

```env
YOUTUBE_API_KEY=AIzaSy...          # Enables YouTube platform
REDDIT_CLIENT_ID=abc123            # Improves Reddit results
REDDIT_CLIENT_SECRET=xyz789
REDDIT_USER_AGENT=SentimentRadar/1.0
```

### 6. Run the app

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## 🎬 Dashboard Overview

> **📺 [Watch the full demo on YouTube](https://youtu.be/ZBAwq79opZk)**

The dashboard includes 5 analysis views:

| Tab | What it shows |
|---|---|
| 🌐 **Overall** | Aggregate sentiment donut + platform grouped bar + radar chart |
| 📊 **By Platform** | Per-platform donut charts with individual breakdown metrics |
| 💬 **Comments** | Raw comments filterable by platform and sentiment with confidence scores |
| ⚖️ **Sentiment Gap** | Positivity score (positive% − negative%) comparison across platforms |
| 🎯 **Deep Dive** | Confidence distribution histogram + volume vs positivity bubble chart + full summary table |

---

## 🔑 API Keys Guide

| Key | Platform | Where to get | Cost |
|---|---|---|---|
| `YOUTUBE_API_KEY` | YouTube | [Google Cloud Console](https://console.cloud.google.com) → YouTube Data API v3 → Credentials | Free · 10k units/day |
| `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` | Reddit | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) → Create App → script | Free |

> **No key needed for:** HackerNews, Dev.to, Mastodon, and Reddit (falls back to Pullpush.io archive automatically).

---

## 🛠 Troubleshooting

<details>
<summary><b>ModuleNotFoundError: No module named 'app'</b></summary>

Run from the project root, not from inside `app/`:
```bash
streamlit run streamlit_app.py
```
</details>

<details>
<summary><b>OSError: Unable to open best_sentiment_model.keras</b></summary>

Model files are missing from `model_files/`. See [Step 4](#4-add-model-files) above.
</details>

<details>
<summary><b>ModuleNotFoundError: No module named 'tf_keras'</b></summary>

```bash
pip install tf-keras==2.17.0
```
</details>

<details>
<summary><b>ModuleNotFoundError: No module named 'keras.preprocessing.text'</b></summary>

Pickle compatibility issue with the tokenizer. Ensure you are using `inference.py` from this repo — it includes a `sys.modules` shim that resolves this automatically.
</details>

<details>
<summary><b>Reddit returns 0 comments</b></summary>

The app automatically falls back to the Pullpush.io archive. If both fail, wait 60 seconds and retry. For consistent fresh results, add Reddit OAuth credentials to `.env`.
</details>

<details>
<summary><b>YouTube returns 0 / shows quota error</b></summary>

Free quota is 10,000 units/day (~100 searches/day). Quota resets at midnight Pacific Time. The app shows a warning in the sidebar when quota is exceeded.
</details>

<details>
<summary><b>LF/CRLF line ending warnings on git add</b></summary>

Run the following to apply the `.gitattributes` line ending rules:
```bash
git rm -r --cached .
git add .
git commit -m "Fix line endings"
```
</details>

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow 2.17.1 + tf-keras 2.17.0 (Keras 2 compatibility layer) |
| NLP | Custom negation-scope preprocessing pipeline + NLTK stopwords |
| ML Utilities | scikit-learn (LabelEncoder, classification metrics) |
| Data Serialisation | joblib (tokenizer + encoder) · Keras H5 (model weights) |
| Dashboard | Streamlit 1.32+ |
| Visualisation | Plotly |
| Data Processing | pandas · numpy |
| Scraping | Python stdlib urllib — no external scraping libraries |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with 🧠 + ☕ by [BhaveshN1015](https://github.com/BhaveshN1015)

[⭐ Star this repo](https://github.com/BhaveshN1015/Sentiment_Radar) if you found it useful

</div>
