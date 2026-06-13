<div align="center">

<img src="https://img.shields.io/badge/📡-Sentiment%20Radar-7c3aed?style=for-the-badge&labelColor=0a0a0f" alt="Sentiment Radar"/>

# Sentiment Radar

**Real-time social media sentiment analysis powered by a custom BiLSTM + Self-Attention deep learning model**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.1-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Keras](https://img.shields.io/badge/tf--keras-2.17.0-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=flat-square)]()

[**📺 Watch Demo**](https://youtu.be/YOUR_VIDEO_ID) · [**📊 Model Details**](#-model-architecture) · [**⚡ Quick Start**](#-quick-start)

<br/>

![Demo GIF](assets/demo.gif)

</div>

---

## 📌 Overview

**Sentiment Radar** is an end-to-end NLP system that scrapes comments from multiple social media platforms on any topic you choose, classifies each comment as **positive**, **neutral**, or **negative** using a custom-trained deep learning model, and presents the results in an interactive dark-theme dashboard with rich visualisations.

Unlike generic sentiment APIs, this model was trained specifically on social media language — handling slang, negation (`"not bad"` → positive), sarcasm markers, and informal text that breaks most off-the-shelf classifiers.

### What it does

- 🔍 **Scrapes on demand** — enter any topic, fetch up to 50 comments per platform in seconds
- 🧠 **Classifies with a custom model** — BiLSTM + Self-Attention trained on 6 social media datasets
- 📊 **Visualises everything** — donut charts, platform radar, sentiment gap bars, confidence histograms
- 💬 **Shows raw comments** — filterable by platform and sentiment with confidence scores
- 📥 **Exports results** — download full analysis as CSV

### Platforms supported

| Platform | Auth required | Status |
|---|---|---|
| Reddit | Optional (OAuth improves results) | ✅ Active |
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

### 🧠 Model Architecture

The core model is a **Bidirectional LSTM with Self-Attention and dual pooling**, trained from scratch on 1.8M+ social media comments aggregated from 6 datasets.

```
Input (seq_len=100)
        │
        ▼
Embedding Layer (vocab=20,000 · dim=128)
        │
SpatialDropout1D(0.3)
        │
        ▼
BiLSTM(128 units, return_sequences=True)
        │
BiLSTM(64 units, return_sequences=True)
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
  Dense(128, ReLU, L2)
  Dropout(0.5)
  Dense(64, ReLU)
  Dropout(0.3)
        │
        ▼
  Dense(3, Softmax)
  [negative · neutral · positive]
```

| Property | Value |
|---|---|
| Training data | 1.8M+ comments (Twitter, Reddit, IMDB, Amazon, YouTube, HackerNews) |
| Vocab size | 20,000 tokens |
| Max sequence length | 100 tokens |
| Negation handling | Custom `_NEG` suffix tagging (`"not good"` → `"not good_NEG"`) |
| Confidence threshold | 50% (below → "uncertain") |
| Saved format | Keras H5 (keras 2.10.0), loaded via tf-keras 2.17 |

### 🔤 Preprocessing Pipeline

A key innovation is the **negation scope tagging** — applied before tokenisation so the model learns that `good` and `good_NEG` are distinct signals:

```
Input:  "I didn't enjoy this at all."
  │
  ├─ 1. Expand contractions  → "I did not enjoy this at all."
  ├─ 2. Negation scope tag   → "I did not enjoy_NEG this_NEG at_NEG all_NEG."
  ├─ 3. Strip non-alpha      → "I did not enjoy_NEG this_NEG at_NEG all_NEG"
  └─ 4. Remove stopwords     → "not enjoy_NEG all_NEG"
          (keeping: not, but, very, can, won't, etc.)

Output: "not enjoy_NEG all_NEG"   → Predicted: NEGATIVE (98.9%)
```

---

## 📁 Project Structure

```
sentiment-radar/
│
├── streamlit_app.py            ← Main dashboard entry point
│
├── app/
│   ├── __init__.py
│   ├── inference.py            ← Model loading, preprocessing, batched prediction
│   └── scraper.py              ← Multi-platform comment scraper
│
├── model_files/                ← Place your 3 model artifacts here
│   ├── best_sentiment_model.keras
│   ├── tokenizerB.joblib
│   └── encoderB.joblib
│
├── nltk_data/                  ← Pre-downloaded NLTK corpora (committed)
│
├── .streamlit/
│   ├── config.toml             ← Dark theme configuration
│   └── secrets.toml            ← API keys (gitignored — never committed)
│
├── requirements.txt            ← Linux / Streamlit Cloud dependencies
├── requirements-windows.txt    ← Windows local dev dependencies
├── setup.sh                    ← Pre-start script for Streamlit Cloud
├── download_nltk.py            ← One-time NLTK data download
├── .env.example                ← Environment variable template
└── .gitignore
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10
- The 3 model files (`best_sentiment_model.keras`, `tokenizerB.joblib`, `encoderB.joblib`)

### 1. Clone and set up

```bash
git clone https://github.com/BhaveshN1015/sentiment-radar.git
cd sentiment-radar
```

### 2. Create virtual environment

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
cp .env.example .env        # Mac/Linux
```

Edit `.env` and add your keys (all optional — app works without them):

```env
YOUTUBE_API_KEY=AIzaSy...          # Enables YouTube platform
REDDIT_CLIENT_ID=abc123            # Improves Reddit results
REDDIT_CLIENT_SECRET=xyz789
```

### 6. Run

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## 🚀 Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub

Make sure your repo includes the model files. Two options:

**Option A — Commit directly** (simplest, files are ~68 MB total, under GitHub's 100 MB limit):
```bash
# In .gitignore, comment out these two lines:
# model_files/*.keras
# model_files/*.joblib

git add model_files/
git commit -m "Add model files"
git push
```

**Option B — Git LFS** (cleaner for large files):
```bash
git lfs install
git lfs track "model_files/*.keras" "model_files/*.joblib"
git add .gitattributes model_files/
git commit -m "Add model files via LFS"
git push
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub
2. Click **New app**
3. Set:
   - **Repository:** `your-username/sentiment-radar`
   - **Branch:** `main`
   - **Main file:** `streamlit_app.py`
4. Click **Advanced settings** → add Secrets:
```toml
YOUTUBE_API_KEY = "AIzaSy..."      # optional
REDDIT_CLIENT_ID = ""              # optional
REDDIT_CLIENT_SECRET = ""          # optional
```
5. Click **Deploy!**

Your app will be live at `https://your-app-name.streamlit.app`

> **⚠️ RAM Note:** Streamlit Cloud free tier provides 1 GB RAM. TensorFlow + model uses ~850–950 MB. If the app crashes on load, click **Reboot app** in the dashboard — it usually recovers on a clean start.

---

## 🎬 Demo

> **📺 [Watch the full demo on YouTube](https://youtu.be/ZBAwq79opZk)**

The dashboard includes 5 analysis views:

| Tab | What it shows |
|---|---|
| 🌐 **Overall** | Aggregate sentiment donut + platform grouped bar + radar chart |
| 📊 **By Platform** | Per-platform donut charts with breakdown metrics |
| 💬 **Comments** | Raw comments filterable by platform and sentiment |
| ⚖️ **Sentiment Gap** | Positivity score comparison across platforms |
| 🎯 **Deep Dive** | Confidence distribution histogram + bubble chart + summary table |

---

## 🔑 API Keys Guide

| Key | Platform | Where to get | Cost |
|---|---|---|---|
| `YOUTUBE_API_KEY` | YouTube | [Google Cloud Console](https://console.cloud.google.com) → YouTube Data API v3 → Credentials | Free (10k units/day) |
| `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` | Reddit | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) → Create App → script | Free |

> Reddit works without credentials via the Pullpush.io archive fallback. YouTube requires a key. HackerNews, Dev.to, and Mastodon need no keys at all.

---

## 🛠 Troubleshooting

<details>
<summary><b>ModuleNotFoundError: No module named 'app'</b></summary>

Run from the project root directory, not from inside `app/`:
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

This is a pickle compatibility issue with the tokenizer. Make sure you are using `inference.py` from this repo — it includes the sys.modules shim that resolves this automatically.
</details>

<details>
<summary><b>Reddit returns 0 comments</b></summary>

Reddit rate-limits unauthenticated scrapers. The app automatically falls back to Pullpush.io archive. If both fail, wait 60 seconds and retry. For consistent results, add Reddit OAuth credentials to `.env`.
</details>

<details>
<summary><b>YouTube returns 0 / quota error</b></summary>

Free quota is 10,000 units/day (100 units per search = ~100 searches/day). Quota resets at midnight Pacific Time. The app shows a warning in the sidebar when quota is exceeded.
</details>

<details>
<summary><b>App crashes on Streamlit Cloud (MemoryError)</b></summary>

Click **Reboot app** in your Streamlit Cloud dashboard. If it persists, the free tier RAM is exhausted — contact Streamlit support to request a resource upgrade for ML apps.
</details>

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow 2.17 + tf-keras 2.17 (Keras 2 compatibility) |
| NLP | Custom preprocessing pipeline + NLTK stopwords |
| Data serialisation | joblib (tokenizer + encoder), H5 (model weights) |
| Dashboard | Streamlit 1.32+ |
| Visualisation | Plotly |
| Data | pandas, numpy |
| Scraping | stdlib urllib (no external scraping libs) |
| Deployment | Streamlit Community Cloud |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with 🧠 + ☕ · [⭐ Star this repo](https://github.com/BhaveshN1015/sentiment-radar) if you found it useful

</div>
