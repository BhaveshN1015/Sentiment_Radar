#!/bin/bash
# setup.sh
# ─────────────────────────────────────────────────────────────
# Streamlit Community Cloud runs this script automatically
# before starting the app. It downloads the NLTK data
# (stopwords corpus) that the inference pipeline needs.
#
# For local use: python download_nltk.py
# ─────────────────────────────────────────────────────────────
set -e
echo ">>> Downloading NLTK data..."
python download_nltk.py
echo ">>> NLTK data ready."
