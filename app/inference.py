"""
inference.py
------------
Loads model / tokenizer / encoder ONCE at module import.
All prediction is fully batched — never one comment at a time.

Keras pickle compatibility fix:
    tokenizerB.joblib was saved with keras 2.10's keras.preprocessing.text.Tokenizer.
    In TF 2.17 + tf-keras, that class lives at tf_keras.preprocessing.text.Tokenizer.
    We inject a sys.modules alias BEFORE joblib unpickles the file so pickle's
    find_class() resolves correctly without any file changes.
"""
from __future__ import annotations

# ── Must be set BEFORE any TF import ─────────────────────────
import os
os.environ["TF_USE_LEGACY_KERAS"]     = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
import sys
import logging
from functools import lru_cache

import nltk


# ── Auto-load .env ────────────────────────────────────────────
def _load_env_inf():
    from pathlib import Path
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip(); val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

_load_env_inf()

# ── NLTK local data path ──────────────────────────────────────
_BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NLTK_DIR  = os.path.join(_BASE, "nltk_data")
if _NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_DIR)

from nltk.corpus import stopwords

# ── tf-keras import ───────────────────────────────────────────
try:
    import tf_keras as keras
    from tf_keras.preprocessing.sequence import pad_sequences
    import tf_keras.preprocessing.text as _keras_text_module
except ImportError:
    import tensorflow.keras as keras
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import tensorflow.keras.preprocessing.text as _keras_text_module

# ── Pickle compatibility shim ─────────────────────────────────
# tokenizerB.joblib was pickled as keras.preprocessing.text.Tokenizer
# We alias the old module paths → tf_keras equivalents so unpickling works.
import types

def _make_alias(real_module, *alias_paths):
    """Register real_module under every alias_path in sys.modules."""
    for path in alias_paths:
        parts = path.split(".")
        # Register every prefix too (keras, keras.preprocessing, …)
        for i in range(1, len(parts) + 1):
            key = ".".join(parts[:i])
            if key not in sys.modules:
                # Create a lightweight stub module for intermediate paths
                stub = types.ModuleType(key)
                sys.modules[key] = stub
        # Point the full path at the real module
        sys.modules[path] = real_module

_make_alias(
    _keras_text_module,
    "keras.preprocessing.text",
    "keras.preprocessing",          # needed for intermediate lookup
)

# Also alias keras.preprocessing.sequence for pad_sequences if needed
try:
    import tf_keras.preprocessing.sequence as _keras_seq_module
except ImportError:
    import tensorflow.keras.preprocessing.sequence as _keras_seq_module

_make_alias(
    _keras_seq_module,
    "keras.preprocessing.sequence",
)

# Alias top-level keras → tf_keras so any remaining keras.* lookups work
if "keras" not in sys.modules or sys.modules["keras"] is not keras:
    sys.modules.setdefault("keras", keras)

# sklearn alias — encoderB.joblib was pickled with sklearn.preprocessing.LabelEncoder
try:
    import sklearn
    import sklearn.preprocessing
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.preprocessing", sklearn.preprocessing)
except ImportError:
    pass  # will fail naturally when joblib tries to load

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR      = os.getenv("MODEL_DIR", os.path.join(_BASE, "model_files"))
MODEL_PATH     = os.path.join(MODEL_DIR, "best_sentiment_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizerB.joblib")
ENCODER_PATH   = os.path.join(MODEL_DIR, "encoderB.joblib")

MAX_LEN              = 100
CONFIDENCE_THRESHOLD = 0.50

# ── Load once ─────────────────────────────────────────────────
logger.info("Loading model from %s ...", MODEL_DIR)
_model     = keras.models.load_model(MODEL_PATH)
_tokenizer = joblib.load(TOKENIZER_PATH)   # now resolves keras.preprocessing.text ✓
_encoder   = joblib.load(ENCODER_PATH)
logger.info("Model ready. Classes: %s", list(_encoder.classes_))

# ── Preprocessing constants ───────────────────────────────────
_KEEP_WORDS: set = {
    "no","nor","not","don","don't","didn","didn't","isn","isn't","aren","aren't",
    "wasn","wasn't","weren","weren't","haven","haven't","hasn","hasn't","hadn",
    "hadn't","won","won't","wouldn","wouldn't","shouldn","shouldn't","couldn",
    "couldn't","mightn","mightn't","mustn","mustn't","needn","needn't","shan",
    "shan't","but","against","very","too","more","most","should","could","would",
    "can","will",
}

_CONTRACTIONS: dict = {
    "won't":"will not","can't":"can not","don't":"do not","doesn't":"does not",
    "didn't":"did not","isn't":"is not","aren't":"are not","wasn't":"was not",
    "weren't":"were not","haven't":"have not","hasn't":"has not","hadn't":"had not",
    "wouldn't":"would not","couldn't":"could not","shouldn't":"should not",
    "mightn't":"might not","mustn't":"must not","needn't":"need not",
    "shan't":"shall not","i'm":"i am","he's":"he is","she's":"she is",
    "it's":"it is","we're":"we are","they're":"they are","i've":"i have",
    "we've":"we have","they've":"they have","i'd":"i would","you'd":"you would",
    "he'd":"he would","i'll":"i will","we'll":"we will","they'll":"they will",
    "that's":"that is",
}

_NEGATION_WORDS     = {"not","no","never","nobody","nothing",
                        "neither","nor","nowhere","hardly","barely","scarcely"}
_CLAUSE_PUNCTUATION = {".","!","?",",",";",":"}


@lru_cache(maxsize=1)
def _stop_words() -> frozenset:
    return frozenset(set(stopwords.words("english")) - _KEEP_WORDS)


def _expand_contractions(text: str) -> str:
    text = text.lower()
    for k, v in _CONTRACTIONS.items():
        text = text.replace(k, v)
    return text


def _apply_negation_scope(text: str) -> str:
    tokens   = re.split(r"(\s+|[.!?,;:])", text)
    negating = False
    result   = []
    for token in tokens:
        if not token.strip():
            result.append(token); continue
        if token in _CLAUSE_PUNCTUATION:
            negating = False; result.append(token)
        elif token.lower() in _NEGATION_WORDS:
            negating = True; result.append(token)
        elif negating:
            result.append(token + "_NEG")
        else:
            result.append(token)
    return "".join(result)


def clean_text(text: str) -> str:
    sw     = _stop_words()
    text   = _expand_contractions(str(text).lower())
    text   = _apply_negation_scope(text)
    review = re.sub(r"[^a-zA-Z_]", " ", text)
    return " ".join(
        w for w in review.split()
        if w not in sw and not (w.endswith("_NEG") and w[:-4] in sw)
    )


# ── Public API ────────────────────────────────────────────────

def predict_batch(texts: list) -> list:
    if not texts:
        return []
    cleaned   = [clean_text(t) for t in texts]
    sequences = _tokenizer.texts_to_sequences(cleaned)
    padded    = pad_sequences(sequences, maxlen=MAX_LEN,
                               padding="post", truncating="post")
    probs_batch = _model.predict(padded, verbose=0)
    results = []
    for probs in probs_batch:
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx]) * 100
        sentiment  = (
            "uncertain" if float(probs[pred_idx]) < CONFIDENCE_THRESHOLD
            else str(_encoder.classes_[pred_idx])
        )
        results.append({
            "sentiment":  sentiment,
            "confidence": round(confidence, 2),
            "probabilities": {
                cls: round(float(p) * 100, 2)
                for cls, p in zip(_encoder.classes_, probs)
            },
        })
    return results


def analyze_platforms(platform_comments: dict) -> dict:
    """
    Input : {"Reddit": ["comment", ...], "HackerNews": [...]}
    Output: {"Reddit": [{"comment":str, "sentiment":str,
                         "confidence":float, "probabilities":dict}, ...]}
    """
    output = {}
    for platform, comments in platform_comments.items():
        if not comments:
            output[platform] = []
            continue
        preds = predict_batch(comments)
        output[platform] = [
            {"comment": c, "sentiment": p["sentiment"],
             "confidence": p["confidence"], "probabilities": p["probabilities"]}
            for c, p in zip(comments, preds)
        ]
        logger.info("[%s] %d comments analysed.", platform, len(comments))
    return output
