"""
download_nltk.py
----------------
Downloads required NLTK corpora.
Run once before first app launch:
    python download_nltk.py

Streamlit Cloud runs this automatically via setup.sh.
"""
import nltk
import os

# Use a writable path that works on both Windows and Linux
nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

for corpus in ["stopwords", "punkt", "punkt_tab"]:
    nltk.download(corpus, download_dir=nltk_data_dir, quiet=True)
    print(f"  ✓ {corpus}")

print(f"\nNLTK data saved to: {nltk_data_dir}")
print("Ready to run: streamlit run streamlit_app.py")
