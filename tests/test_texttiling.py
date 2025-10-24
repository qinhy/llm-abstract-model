# TextTiling → hierarchical tree with automatic depth & labels
import re, json, os
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import re
from collections import Counter
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def make_corpus_stats(segs: List[str]):
    # binary term presence per segment → document frequencies
    v = TfidfVectorizer(max_features=8000, ngram_range=(1,2), stop_words="english", use_idf=False, norm=None)
    X = v.fit_transform(segs)  # term counts
    df = (X > 0).sum(axis=0).A1  # document frequency per term
    return v, df, len(segs)

def label_node_c_tfidf(node, segs, vectorizer, df, N, top_n=5):
    idxs = gather_leaf_indices(node)
    if not idxs:
        node["title"] = "∅"
        return node
    # class doc: sum counts over this node’s segments
    X = vectorizer.transform([segs[i] for i in idxs])
    tf_c = X.sum(axis=0).A1
    # c-TF-IDF ~ (tf_c / |class|) * log(1 + N / df)
    scores = (tf_c / max(1, len(idxs))) * np.log1p(N / np.maximum(1, df))
    top = scores.argsort()[::-1][:top_n]
    terms = [vectorizer.get_feature_names_out()[i] for i in top]
    node["title"] = ", ".join(terms)
    # recurse
    if "children" in node:
        node["children"] = [label_node_c_tfidf(ch, segs, vectorizer, df, N, top_n) for ch in node["children"]]
    return node

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", re.UNICODE)

def split_paragraphs(text: str) -> List[str]:
    # Paragraphs = blocks separated by >=1 blank line
    paras = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paras if p.strip()]

def tokenize_words(s: str) -> List[str]:
    # Lowercased word tokens; you can add stemming if desired
    return [m.group(0).lower() for m in _WORD_RE.finditer(s)]

def cosine_from_counters(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    # dot
    dot = 0.0
    (ka, kb) = (a.keys(), b.keys())
    if len(ka) < len(kb):
        for k, va in a.items():
            vb = b.get(k)
            if vb:
                dot += va * vb
    else:
        for k, vb in b.items():
            va = a.get(k)
            if va:
                dot += va * vb
    # norms
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

def moving_average(xs: List[float], w: int) -> List[float]:
    if w <= 1 or len(xs) < w:
        return xs[:]
    out = []
    k = w // 2
    for i in range(len(xs)):
        lo = max(0, i - k)
        hi = min(len(xs), i + k + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out

# --------- Core TextTiling ---------
def texttiling_tokenize(
    text: str,
    w: int = 20,             # token sequence size (pseudo-sentence)
    k: int = 10,             # block size in sequences
    smooth_width: int = 2,   # smoothing window over similarity curve
    smooth_rounds: int = 1,  # times to apply smoothing
    boundary_policy: str = "percentile",  # "percentile" or "z"
    boundary_param: float = 70.0,         # percentile (if "percentile") or z-threshold (if "z")
    min_segment_paras: int = 1            # avoid tiny segments by coalescing cuts
) -> List[str]:
    """
    TextTiling-style segmentation that snaps boundaries to paragraph breaks.

    Returns: list[str] coherent segments.
    """
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    # Tokenize paragraphs and compute cumulative token counts per paragraph
    para_tokens = [tokenize_words(p) for p in paragraphs]
    para_lengths = [len(toks) for toks in para_tokens]
    total_tokens = sum(para_lengths)
    if total_tokens == 0:
        return ["\n\n".join(paragraphs)]

    cum_para_tokens = []
    running = 0
    for L in para_lengths:
        running += L
        cum_para_tokens.append(running)

    # Build token sequences (flatten words)
    words: List[str] = [tok for toks in para_tokens for tok in toks]
    sequences: List[List[str]] = [
        words[i : i + w] for i in range(0, len(words), w)
    ]
    if len(sequences) < 2:
        return ["\n\n".join(paragraphs)]

    # Precompute term counters per sequence
    seq_counters = [Counter(seq) for seq in sequences]

    # Compute block similarity curve between adjacent blocks of k sequences
    sims: List[float] = []
    # Define valid boundary indices in "sequence space"
    # boundary i is between sequences i and i+1
    for i in range(len(sequences) - 1):
        # left block = sequences [i-k+1..i], right block = sequences [i+1..i+k]
        left_lo = max(0, i - k + 1)
        left_hi = i + 1
        right_lo = i + 1
        right_hi = min(len(sequences), i + 1 + k)
        if left_hi - left_lo < 1 or right_hi - right_lo < 1:
            sims.append(0.0)
            continue
        left = sum(seq_counters[left_lo:left_hi], Counter())
        right = sum(seq_counters[right_lo:right_hi], Counter())
        sims.append(cosine_from_counters(left, right))

    # Smooth similarity curve to reduce noise
    for _ in range(max(0, smooth_rounds)):
        sims = moving_average(sims, max(1, smooth_width))

    # Compute depth scores at each boundary i (valley strength)
    depths = depth_scores(sims)

    # Choose boundaries by policy
    idxs = pick_boundaries(depths, policy=boundary_policy, param=boundary_param)

    # Map sequence boundary i → token boundary index (i+1)*w
    # Then snap to nearest paragraph boundary (on or after that point)
    cut_paras = set()
    for i in idxs:
        token_boundary = (i + 1) * w
        # find first paragraph whose cumulative token count >= token_boundary
        p = first_para_index_at_or_after_token(cum_para_tokens, token_boundary)
        if p is not None and 0 < p < len(paragraphs):
            cut_paras.add(p)

    # Enforce min_segment_paras by merging too-small slices
    cut_list = sorted(cut_paras)
    cut_list = coalesce_small_slices(cut_list, len(paragraphs), min_segment_paras)

    # Build final segments by slicing paragraphs
    segments: List[str] = []
    prev = 0
    for cp in cut_list:
        seg = "\n\n".join(paragraphs[prev:cp]).strip()
        if seg:
            segments.append(seg)
        prev = cp
    tail = "\n\n".join(paragraphs[prev:]).strip()
    if tail:
        segments.append(tail)
    return segments

def depth_scores(sims: List[float]) -> List[float]:
    """Depth at i = (left_peak - sims[i]) + (right_peak - sims[i]) using nearest local maxima."""
    n = len(sims)
    if n == 0:
        return []
    # Identify local maxima indices
    max_idxs = local_maxima(sims)
    if not max_idxs:
        return [0.0] * n

    depths = [0.0] * n
    for i in range(n):
        # find nearest maxima to left and right
        lm = max([m for m in max_idxs if m < i], default=None)
        rm = min([m for m in max_idxs if m > i], default=None)
        if lm is None or rm is None:
            depths[i] = 0.0
        else:
            depths[i] = max(0.0, (sims[lm] - sims[i]) + (sims[rm] - sims[i]))
    return depths

def local_maxima(xs: List[float]) -> List[int]:
    n = len(xs)
    out = []
    for i in range(n):
        left = xs[i - 1] if i - 1 >= 0 else float("-inf")
        right = xs[i + 1] if i + 1 < n else float("-inf")
        if xs[i] >= left and xs[i] >= right:
            out.append(i)
    return out

def pick_boundaries(depths: List[float], policy: str, param: float) -> List[int]:
    if not depths:
        return []
    if policy == "percentile":
        # Keep positions whose depth is >= chosen percentile
        import numpy as np
        thr = float(np.percentile(depths, param))
        return [i for i, d in enumerate(depths) if d >= thr]
    elif policy == "z":
        # Keep positions with (depth - mean) / std >= param
        import math
        mean = sum(depths) / len(depths)
        var = sum((d - mean) ** 2 for d in depths) / max(1, len(depths) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        if std == 0.0:
            return []
        return [i for i, d in enumerate(depths) if (d - mean) / std >= param]
    else:
        # Fallback: top-k =  √N boundaries
        k = max(1, int(len(depths) ** 0.5))
        order = sorted(range(len(depths)), key=lambda i: depths[i], reverse=True)
        return order[:k]

def first_para_index_at_or_after_token(cum_para_tokens: List[int], token_idx: int):
    # binary search
    lo, hi = 0, len(cum_para_tokens) - 1
    ans = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if cum_para_tokens[mid] >= token_idx:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

def coalesce_small_slices(cuts: List[int], n_paras: int, min_len: int) -> List[int]:
    """Merge boundaries so that every segment has at least min_len paragraphs."""
    if not cuts or min_len <= 1:
        return cuts
    out = []
    prev = 0
    for c in cuts:
        if (c - prev) < min_len:
            # skip this cut; merge with next
            continue
        out.append(c)
        prev = c
    # check tail
    if (n_paras - (out[-1] if out else 0)) < min_len:
        # drop last cut if tail would be too short
        out = out[:-1]
    return out

# -------- 1) Segment with TextTiling --------
def texttiling_segments(
    text: str,
    w: int = 20,
    k: int = 10,
    smoothing_width: int = 2,
    smoothing_rounds: int = 2,
    # map NLTK-ish name to our policy:
    cutoff_policy: str = "percentile",   # "percentile" or "z"
    cutoff_param: float = 70.0,          # 70 => keep deepest 30% valleys
    min_segment_paras: int = 1
) -> List[str]:
    """
    Returns coherent multi-paragraph segments (leaves).
    """
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    segs = texttiling_tokenize(
        text,
        w=w,
        k=k,
        smooth_width=smoothing_width,
        smooth_rounds=smoothing_rounds,
        boundary_policy=cutoff_policy,
        boundary_param=cutoff_param,
        min_segment_paras=min_segment_paras,
    )
    return [re.sub(r'\s+', ' ', s).strip() for s in segs if s and s.strip()]

# -------- 2) Vectors for clustering & labeling --------
def tfidf_vectors(docs: List[str]):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
    X = vec.fit_transform(docs)
    return X, vec

# -------- 3) Auto-depth hierarchical induction (quality-driven splits) --------
def try_bisect(X_sub, min_leaf=6, min_sil=0.02, root=False):
    if X_sub.shape[0] < (min_leaf * 2):
        return False, None, None
    try:
        model = AgglomerativeClustering(n_clusters=2, linkage="average", metric="cosine")
    except TypeError:  # scikit-learn <1.2
        model = AgglomerativeClustering(n_clusters=2, linkage="average", affinity="cosine")
    labels = model.fit_predict(X_sub.toarray())
    if min(labels.sum(), (labels == 0).sum()) < min_leaf:
        return False, None, None
    score = silhouette_score(X_sub.toarray(), labels, metric="cosine")
    thresh = (min_sil * 0.5) if root else min_sil   # be looser at the root
    return (score >= thresh), labels, score

def build_tree(X, indices, depth=0, min_leaf=6, max_depth=8, min_sil=0.02):
    if depth >= max_depth or indices.size < (min_leaf * 2):
        return {"leaf_indices": indices.tolist()}
    improves, labels, score = try_bisect(X[indices], min_leaf=min_leaf, min_sil=min_sil, root=(depth==0))
    if not improves:
        return {"leaf_indices": indices.tolist(), "note": f"stopped: sil<={min_sil} (score={score})"}
    left = indices[labels == 0]; right = indices[labels == 1]
    return {
        "children": [
            build_tree(X, left,  depth+1, min_leaf, max_depth, min_sil),
            build_tree(X, right, depth+1, min_leaf, max_depth, min_sil),
        ],
        "split_silhouette": float(score)
    }

# -------- 4) c-TF-IDF-style labeling for each node --------
def label_node(node: Dict[str, Any], segs: List[str], top_n=5):
    if "leaf_indices" in node:
        texts = [segs[i] for i in node["leaf_indices"]]
        node["title"] = summarize_phrases(texts, top_n=top_n)
        return node
    # internal node: gather all descendant texts and label
    indices = gather_leaf_indices(node)
    node["title"] = summarize_phrases([segs[i] for i in indices], top_n=top_n)
    # recurse
    node["children"] = [label_node(ch, segs, top_n) for ch in node["children"]]
    return node

def gather_leaf_indices(node: Dict[str, Any]) -> List[int]:
    if "leaf_indices" in node:
        return node["leaf_indices"]
    out = []
    for ch in node.get("children", []):
        out.extend(gather_leaf_indices(ch))
    return out

def summarize_phrases(docs: List[str], top_n=5) -> str:
    """
    Simple c-TF-IDF-style label: compare this group vs. a small background built from itself.
    For stronger labels, compare against sibling clusters; kept minimal here.
    """
    if len(docs) == 0:
        return "∅"
    # Collapse docs to one “class doc” & compare inside-document distribution
    text = " ".join(docs)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
    X = vec.fit_transform([text])
    scores = X.toarray().ravel()
    top_ids = scores.argsort()[::-1][:top_n]
    terms = [vec.get_feature_names_out()[i] for i in top_ids]
    return ", ".join(terms)

# -------- 5) Public API --------
def texttiling_tree(text: str,
                    ttt_w=20, ttt_k=10,
                    min_leaf=6, max_depth=8,
                    top_n_labels=5,
                    cutoff_policy="percentile",
                    cutoff_param=70.0,
                    smoothing_width=2,
                    smoothing_rounds=2,
                    min_segment_paras=1) -> Dict[str, Any]:

    segs = texttiling_segments(
        text, w=ttt_w, k=ttt_k,
        smoothing_width=smoothing_width, smoothing_rounds=smoothing_rounds,
        cutoff_policy=cutoff_policy, cutoff_param=cutoff_param,
        min_segment_paras=min_segment_paras
    )

    X, _ = tfidf_vectors(segs)
    root = build_tree(X, np.arange(len(segs)), min_leaf=min_leaf, max_depth=max_depth)

    vec_corpus, df, N = make_corpus_stats(segs)
    root = label_node_c_tfidf(root, segs, vec_corpus, df, N, top_n=top_n_labels)
    # root = label_node(root, segs, top_n=top_n_labels)

    # Attach segment previews at leaves (your “content leaves”)
    def attach_leaf_content(node):
        if "leaf_indices" in node:
            node["leaves"] = [
                {"i": int(i), "preview": segs[i][:80] + ("…" if len(segs[i]) > 80 else "")}
                for i in node["leaf_indices"]
            ]
            del node["leaf_indices"]
        else:
            node["children"] = [attach_leaf_content(ch) for ch in node["children"]]
        return node

    return attach_leaf_content({"title": root.get("title", "root"),
                                "children": root.get("children", []),
                                **{k:v for k,v in root.items() if k not in ("children","title")}})
def node_size(n):
    if "leaves" in n:
        return len(n["leaves"])
    return sum(node_size(c) for c in n.get("children", []))

def pretty_print_tree(
    node,
    *,
    max_depth=None,
    show_leaf_previews=False,
    max_preview_per_node=3,
    truncate=80,
    _prefix="",
    _is_last=True,
    _depth=0,
):
    """Prints an ASCII tree like:
    root [42]
    ├─ Chapter 1 [10] (sil=0.081)
    │  └─ Scene A [4]
    └─ Chapter 2 [32] ...
    """
    # header line
    branch = "└─ " if _is_last else "├─ "
    line_prefix = _prefix + (branch if _depth > 0 else "")
    title = node.get("title", "∅")
    size = node_size(node)
    sil = node.get("split_silhouette")
    note = node.get("note")

    parts = [f"{title}  [{size}]"]
    if sil is not None:
        parts.append(f"(sil={sil:.3f})")
    if note:
        parts.append(f"{note}")
    print(line_prefix + " ".join(parts))

    # stop at depth
    if max_depth is not None and _depth >= max_depth:
        return

    # children or leaves
    children = node.get("children", [])
    if children:
        new_prefix = _prefix + ("   " if _is_last else "│  ")
        for i, ch in enumerate(children):
            pretty_print_tree(
                ch,
                max_depth=max_depth,
                show_leaf_previews=show_leaf_previews,
                max_preview_per_node=max_preview_per_node,
                truncate=truncate,
                _prefix=new_prefix,
                _is_last=(i == len(children) - 1),
                _depth=_depth + 1,
            )
    elif show_leaf_previews:
        bullets_prefix = _prefix + ("   " if _is_last else "│  ")
        for i, leaf in enumerate(node.get("leaves", [])[:max_preview_per_node]):
            prev = leaf.get("preview", "")
            prev = (prev[:truncate] + "…") if len(prev) > truncate else prev
            print(bullets_prefix + f"• #{leaf.get('i', i)}: {prev}")

# -------- Example usage --------
if __name__ == "__main__":
    # 1) Put the full plaintext of the book into The Adventures of Sherlock Holmes.txt (double newlines between paragraphs).
    with open("The Adventures of Sherlock Holmes.txt", "r", encoding="utf-8") as f:
        book = f.read()

    tree = texttiling_tree(book, 
                        ttt_w=40,          # was 20
                        ttt_k=15,          # was 10
                        min_leaf=4,        # allow smaller clusters to form
                        max_depth=6,
                        top_n_labels=5,
                        cutoff_policy="percentile",
                        cutoff_param=85.0, # keep only deepest 15% valleys
                        smoothing_width=3,
                        smoothing_rounds=2,
                        min_segment_paras=2)
    print(json.dumps(tree, ensure_ascii=False, indent=2))
    pretty_print_tree(tree, max_depth=3, show_leaf_previews=True, max_preview_per_node=2)

