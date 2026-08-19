"""
Embedding-based clip similarity detection.
Uses sentence-transformers to find and remove near-duplicate clips.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _get_model():
    """Lazy-load the sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def compute_clip_embeddings(clips: List[Dict[str, Any]]) -> np.ndarray:
    """Compute embeddings for clip descriptions.

    Args:
        clips: List of clip dicts with 'reasoning' and/or 'text' fields.

    Returns:
        numpy array of shape (n_clips, embedding_dim).
    """
    model = _get_model()

    texts = []
    for clip in clips:
        parts = []
        if clip.get("reasoning"):
            parts.append(clip["reasoning"])
        if clip.get("text"):
            parts.append(clip["text"])
        texts.append(" ".join(parts) if parts else f"clip at {clip.get('start_time', 0)}s")

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_similar_pairs(
    embeddings: np.ndarray,
    threshold: float = 0.75,
) -> List[Tuple[int, int, float]]:
    """Find pairs of clips with similarity above threshold.

    Args:
        embeddings: Array of shape (n_clips, dim).
        threshold: Similarity threshold (0-1).

    Returns:
        List of (idx_a, idx_b, similarity_score) tuples, sorted by similarity desc.
    """
    n = len(embeddings)
    pairs: List[Tuple[int, int, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                pairs.append((i, j, sim))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def deduplicate_clips(
    clips: List[Dict[str, Any]],
    similarity_threshold: float = 0.85,
    min_clips: int = 1,
) -> List[Dict[str, Any]]:
    """Remove near-duplicate clips based on embedding similarity.

    When two clips are similar, keeps the one with higher virality_score.
    Won't remove clips below min_clips count.

    Args:
        clips: List of clip dicts.
        similarity_threshold: Cosine similarity threshold for dedup (0-1).
        min_clips: Minimum clips to keep (won't dedup below this).

    Returns:
        Deduplicated list of clips.
    """
    if len(clips) <= min_clips:
        return clips

    print(f"--- Computing clip embeddings for similarity check ({len(clips)} clips) ---")
    embeddings = compute_clip_embeddings(clips)

    print(f"--- Finding similar pairs (threshold={similarity_threshold}) ---")
    pairs = find_similar_pairs(embeddings, threshold=similarity_threshold)

    if not pairs:
        print("  No similar clips found.")
        return clips

    # Mark lower-scoring clips for removal, but respect min_clips
    to_remove = set()
    for idx_a, idx_b, sim in pairs:
        if idx_a in to_remove or idx_b in to_remove:
            continue  # Already removing one of them

        # Stop if we'd go below min_clips
        if len(clips) - len(to_remove) <= min_clips:
            break

        score_a = clips[idx_a].get("virality_score", 0)
        score_b = clips[idx_b].get("virality_score", 0)

        if score_a >= score_b:
            to_remove.add(idx_b)
            print(f"  Removing clip {idx_b} (similar to {idx_a}, sim={sim:.2f})")
        else:
            to_remove.add(idx_a)
            print(f"  Removing clip {idx_a} (similar to {idx_b}, sim={sim:.2f})")

    result = [clip for i, clip in enumerate(clips) if i not in to_remove]
    print(f"  Kept {len(result)}/{len(clips)} clips after deduplication.")
    return result
