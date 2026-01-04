"""Extract YARA signatures from the Barrikada dataset.

This script is intentionally simple for this project:
- Input is always `datasets/barrikada.csv`.
- Output is always YARA rules under `core/layer_b/signatures/extracted/`.

Dataset requirements:
- Column `label`: 0 = SAFE, 1 = MALICIOUS
- Text column: `prompt` (preferred) or `text` (fallback)

Outputs:
- `core/layer_b/signatures/extracted/safe_allow_signatures.yar`
- `core/layer_b/signatures/extracted/malicious_block_high_signatures.yar`
"""

import re
import sys
import hashlib
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.PatternStats import PatternStats


_WS_RE = re.compile(r"\s+") # matches any whitespace sequence
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]") #matches char that is not lowercase, letter or whitespace

# paths
DATASET_CSV = Path("datasets/barrikada.csv")
OUTDIR = Path("core/layer_b/signatures/extracted")

# SAFE signature selection
SAFE_TOP_K = 500 #most frequest n-grams
SAFE_MIN_SUPPORT = 20 #drop anything seen fewer that 20 samples

# SAFE allow-listing selection (must be very safe if we use it for early termination)
SAFE_ALLOW_PRECISION_THRESHOLD = 0.995  # safe/(safe+malicious)
SAFE_ALLOW_MAL_DF_CAP = 1  # appears in <= this many malicious samples

# MALICIOUS signature selection
MAL_MIN_SUPPORT = 20
MAL_PRECISION_THRESHOLD = 0.95 #have at least 95% (malicious / total hits)
MAL_SAFE_DF_CAP = 2 #but seen in at most 2 safe samples


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()  # lowercase
    text = _NON_ALNUM_RE.sub(" ", text)  # remove punctuation
    text = _WS_RE.sub(" ", text).strip()  # collapse whitespace
    return text

def _make_vectorizers() -> Tuple[CountVectorizer, CountVectorizer]:
    """
    Helper that creates two CountVectorizers: one for word n-grams and one for character n-grams.
    
    :return: Description
    :rtype: Tuple[CountVectorizer, CountVectorizer]
    """
    # Token n-grams (1-4)
    word_vec = CountVectorizer(
        analyzer="word",
        ngram_range=(1, 4),
        binary=True, #a feature is either present or absent in a document. 
        min_df=2,
        max_features=50000,
        lowercase=False,
        preprocessor=None,
        tokenizer=str.split,
        token_pattern=None, # "dont do your own preprocessing plz", I said
    )

    # Character n-grams (3-8)
    char_vec = CountVectorizer(
        analyzer="char",
        ngram_range=(3, 8),
        min_df=2,
        max_features=50000,
        lowercase=False,
        binary=True,

    )

    return word_vec, char_vec

# Return document frequency (DF) per feature.
def _doc_freq(X) -> np.ndarray:
    # DF(feature) = number of documents where feature count > 0
    return np.asarray(X.getnnz(axis=0)).ravel().astype(int) #using getmz instead of sum to be robust even if binary changes.


_STOPWORDS = set(ENGLISH_STOP_WORDS)

#no unigrams and no all-stopword phrases
def _is_stopwordy_ngram(pattern: str) -> bool:
    tokens = pattern.split()
    if not tokens:
        return True
    # Disallow unigrams for allowlisting; they are almost always too generic.
    if len(tokens) < 2:
        return True
    return all(t in _STOPWORDS for t in tokens)


def build_safe_signatures(
    texts_norm: List[str],
    labels: np.ndarray,
) -> List[dict]:
    """Build high precision SAFE allow signatures.

    These rules allow early termination (most prompts are SAFE, so we shouldnt waste compute passing them to ML layer etc..)
    They must be common in SAFE and extremely rare in MALICIOUS.
    """
    #split the dataset into SAFE and MALICIOUS masks
    y = labels.astype(int)
    safe_mask = y == 0
    mal_mask = y == 1

    # safe_mask = []
    # mal_mask = []

    # for label in y:
    #     if label == 0:
    #         safe_mask.append(True)
    #         mal_mask.append(False)
    #     else:
    #         safe_mask.append(False)
    #         mal_mask.append(True)

    word_vec, _ = _make_vectorizers() #we only need word n-grams for safe signatures
    Xw = word_vec.fit_transform(texts_norm) #learn vocab from texts
    vocab = word_vec.get_feature_names_out() #get actual n-gram strings

    # "in how many safe documents does this n-gram appear"
    safe_df = _doc_freq(Xw[safe_mask]) # type: ignore  
    mal_df = _doc_freq(Xw[mal_mask])  # type: ignore

    # Select candidates
    candidates: List[PatternStats] = []
    for pattern, s_df_i, m_df_i in zip(vocab, safe_df, mal_df):
        # pattern = str(pat)
        # s_df_i = int(s_df)
        # m_df_i = int(m_df)

        #must appear in at least SAFE_MIN_SUPPORT safe samples (o.w too rare)
        if s_df_i < SAFE_MIN_SUPPORT:
            continue
        #must appear in at most SAFE_ALLOW_MAL_DF_CAP malicious samples (o.w not safe enough)
        if m_df_i > SAFE_ALLOW_MAL_DF_CAP:
            continue
        if _is_stopwordy_ngram(pattern):
            continue

        #how often it appears in SAFE vs total appearances
        precision_safe = s_df_i / (s_df_i + m_df_i) if (s_df_i + m_df_i) else 0.0
        if precision_safe < SAFE_ALLOW_PRECISION_THRESHOLD:
            continue

        candidates.append(PatternStats(pattern, s_df_i, m_df_i))

    candidates.sort(key=lambda x: (x.mal_precision, x.mal_df, -len(x.pattern)), reverse=True)
    candidates = candidates[:SAFE_TOP_K]

    out: List[dict] = []
    for c in candidates:
        out.append(
            {
                "pattern": c.pattern,
                "support": c.safe_df,
                "malicious_support": c.mal_df,
                "safe_precision": float(round(c.safe_precision, 4)),
                "type": "allow-safe",
            }
        )
    return out


def build_malicious_signatures(
    texts_norm: List[str],
    labels: np.ndarray,
) -> List[dict]:
    word_vec, char_vec = _make_vectorizers()

    # Fit on all docs so we can compare safe vs malicious support
    Xw = word_vec.fit_transform(texts_norm)
    Xc = char_vec.fit_transform(texts_norm)

    y = labels.astype(int)
    safe_mask = y == 0
    mal_mask = y == 1

    w_vocab = word_vec.get_feature_names_out()
    c_vocab = char_vec.get_feature_names_out()

    w_safe_df = _doc_freq(Xw[safe_mask])  # type: ignore
    w_mal_df = _doc_freq(Xw[mal_mask])  # type: ignore

    c_safe_df = _doc_freq(Xc[safe_mask])  # type: ignore
    c_mal_df = _doc_freq(Xc[mal_mask])  # type: ignore

    stats: List[PatternStats] = [] # or stats: List[PatternStats] = [] for automcomplete. I didnt cus I want readable code

    for pat, s_df, m_df in zip(w_vocab, w_safe_df, w_mal_df):
        stats.append(PatternStats(pattern=str(pat), safe_df=int(s_df), mal_df=int(m_df)))

    for pat, s_df, m_df in zip(c_vocab, c_safe_df, c_mal_df):
        stats.append(PatternStats(pattern=str(pat), safe_df=int(s_df), mal_df=int(m_df)))

    selected: List[PatternStats] = []
    for st in stats:
        if st.mal_df < MAL_MIN_SUPPORT:
            continue
        if st.mal_precision < MAL_PRECISION_THRESHOLD:
            continue
        if st.safe_df > MAL_SAFE_DF_CAP:
            continue
        # Keep patterns reasonably interpretable
        if len(st.pattern) < 3:
            continue
        selected.append(st)

    selected.sort(key=lambda s: (s.mal_precision, s.mal_df, -len(s.pattern)), reverse=True)

    out = []
    for st in selected:
        out.append(
            {
                "pattern": st.pattern,
                "precision": float(round(st.mal_precision, 4)),
                "support": int(st.mal_df),
                "safe_support": int(st.safe_df),
                "type": "block-high",
            }
        )
    return out


def load_dataset(csv_path: Path) -> Tuple[pd.Series, np.ndarray]:
    df = pd.read_csv(csv_path)

    texts = df["text"].fillna("")
    labels = df["label"].astype(int).to_numpy()
    return texts, labels


def _yara_escape_str_literal(s: str) -> str:
    # YARA double-quoted string
    return s.replace('\\', r'\\').replace('"', r'\\"')


def _yara_word_ngram_regex(pattern: str) -> str:
    # Designed for normalized text (lower, punctuation removed, whitespace collapsed).
    tokens = pattern.split()
    # IMPORTANT: this string is written verbatim into a YARA `/.../` regex.
    # Emit `\s` (not `\\s`) so YARA interprets it as whitespace.
    body = "\\s+".join(re.escape(t) for t in tokens)
    return f"(^|\\s){body}(\\s|$)"


def _pattern_to_yara_string(pattern: str) -> Tuple[str, str]:
    """Return (identifier, yara_string_decl).

    For readability we emit exactly one string per rule:
    - 1–4 token word n-grams: regex with whitespace boundaries (normalized text)
    - otherwise: plain string literal
    """
    tokens = pattern.split()
    if 1 <= len(tokens) <= 4 and all(tok.isalnum() for tok in tokens):
        regex = _yara_word_ngram_regex(pattern)
        return "$re", f"$re = /{regex}/ nocase"

    lit = _yara_escape_str_literal(pattern)
    return "$s", f"$s = \"{lit}\" nocase"


def _yara_meta_value(v: Any) -> str:
    """Render a YARA meta value.

    - integers are emitted as integers
    - floating point values are emitted as strings (YARA does not support floats?!?!)
    - everything else is emitted as a double-quoted string
    """
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"\"{v}\""
    return f"\"{_yara_escape_str_literal(str(v))}\""


def _pattern_digest(pattern: str) -> str:
    # Stable short identifier to make rule names deterministic across runs.
    h = hashlib.blake2b(pattern.encode("utf-8"), digest_size=6)
    return h.hexdigest()


def _render_rule(
    rule_name: str,
    tags: List[str],
    pattern: str,
    ident: str,
    decl: str,
    meta: List[Tuple[str, Any]],
) -> str:
    tag_str = f" : {' '.join(tags)}" if tags else ""

    lines: List[str] = []
    lines.append(f"rule {rule_name}{tag_str} {{")
    lines.append("    meta:")
    lines.append(f"        pattern = \"{_yara_escape_str_literal(pattern)}\"")

    for k, v in meta:
        lines.append(f"        {k} = {_yara_meta_value(v)}")

    lines.append("    strings:")
    lines.append(f"        {decl}")
    lines.append("    condition:")
    lines.append(f"        {ident}")
    lines.append("}")
    return "\n".join(lines)


def write_yara_rules(
    out_path: Path,
    rule_prefix: str,
    signatures: List[dict],
    meta_keys: List[str],
):
    lines: List[str] = []
    lines.append("/*")
    lines.append("  AUTO-GENERATED FILE")
    lines.append("  Generated by: scripts/extract_signature_patterns.py")
    lines.append("  Source: datasets/barrikada.csv")
    lines.append("*/")
    lines.append("")

    for idx, sig in enumerate(signatures, start=1):
        pattern = str(sig["pattern"]).strip()
        if not pattern:
            continue

        digest = _pattern_digest(pattern)
        rule_name = f"{rule_prefix}{digest}_{idx:04d}"
        ident, decl = _pattern_to_yara_string(pattern)

        tags = ["extracted"]
        meta: List[Tuple[str, Any]] = []
        for k in meta_keys:
            if k in sig:
                meta.append((k, sig[k]))

        lines.append(
            _render_rule(
                rule_name=rule_name,
                tags=tags,
                pattern=pattern,
                ident=ident,
                decl=decl,
                meta=meta,
            )
        )
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    texts, labels = load_dataset(DATASET_CSV)
    texts_norm = [normalize_text(t) for t in texts.tolist()]

    n_safe = int((labels == 0).sum())
    n_mal = int((labels == 1).sum())
    print(f"Loaded {len(labels)} samples (safe={n_safe}, malicious={n_mal})")

    safe_signatures = build_safe_signatures(texts_norm, labels)

    malicious_signatures = build_malicious_signatures(texts_norm, labels)

    safe_yar = OUTDIR / "safe_allow_signatures.yar"
    mal_yar = OUTDIR / "malicious_block_high_signatures.yar"

    write_yara_rules(
        safe_yar,
        rule_prefix="SAFE_ALLOW_",
        signatures=safe_signatures,
        meta_keys=["safe_precision", "support", "malicious_support", "type"],
    )
    write_yara_rules(
        mal_yar,
        rule_prefix="DATA_HIGH_",
        signatures=malicious_signatures,
        meta_keys=["precision", "support", "safe_support", "type"],
    )

    print(f"Wrote {len(safe_signatures)} SAFE YARA rules -> {safe_yar}")
    print(f"Wrote {len(malicious_signatures)} MALICIOUS YARA rules -> {mal_yar}")


if __name__ == "__main__":
    main()
