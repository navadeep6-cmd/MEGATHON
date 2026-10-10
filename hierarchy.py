"""Category graph utilities (standard library only).

Commons categories form a directed graph that is a DAG in intent but has
cycles in practice, so every traversal is cycle-safe and depth-limited.

Editors file images under the *most specific* applicable category, so the
post-processor suppresses ancestors of a selected category, while keeping a
high-scoring ancestor when no descendant scores well.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple


class CategoryGraph:
    """Child -> parents map with cached, depth-limited ancestor sets."""

    def __init__(self, parents: Dict[str, Iterable[str]], max_depth: int = 3):
        self.parents: Dict[str, List[str]] = {c: list(ps) for c, ps in parents.items()}
        self.max_depth = max_depth
        self._cache: Dict[str, FrozenSet[str]] = {}

    def ancestors(self, name: str) -> FrozenSet[str]:
        """All ancestors up to max_depth (excluding the category itself)."""
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        seen = set()
        queue = deque([(name, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= self.max_depth:
                continue
            for parent in self.parents.get(node, ()):
                if parent == name or parent in seen:
                    continue
                seen.add(parent)
                queue.append((parent, depth + 1))
        result = frozenset(seen)
        self._cache[name] = result
        return result

    def is_ancestor(self, maybe_ancestor: str, of: str) -> bool:
        return maybe_ancestor in self.ancestors(of)

    def related(self, a: str, b: str) -> bool:
        """True if a is an ancestor or a descendant of b (within max_depth)."""
        return a != b and (self.is_ancestor(a, b) or self.is_ancestor(b, a))

    def ancestors_of_set(self, names: Iterable[str]) -> FrozenSet[str]:
        out = set()
        for n in names:
            out |= self.ancestors(n)
        return frozenset(out)

    def __len__(self) -> int:
        return len(self.parents)


def rerank_with_hierarchy(
    ranked: Sequence[Tuple[str, float]],
    graph: CategoryGraph,
    k: int,
    min_relative_score: float = 0.5,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Most-specific-category post-processing.

    ranked: (category, score) sorted by descending score.
    Walks the list and builds the top-k selection:
      * a category that is an ancestor of an already-selected (higher-scoring)
        category is suppressed;
      * a category that is a descendant of a selected ancestor replaces that
        ancestor if its score is at least min_relative_score * ancestor score;
        otherwise both are kept (the ancestor is the safer suggestion).
    Returns (full_reranked_list, suppressed). The full list is
    selected + untouched remainder + suppressed, so ranking metrics can still
    be computed on it.
    """
    selected: List[Tuple[str, float]] = []
    suppressed: List[Tuple[str, float]] = []
    consumed = 0
    for cat, score in ranked:
        if len(selected) >= k:
            break
        consumed += 1
        if any(cat in graph.ancestors(s) for s, _ in selected):
            suppressed.append((cat, score))
            continue
        cat_anc = graph.ancestors(cat)
        keep: List[Tuple[str, float]] = []
        for s, s_score in selected:
            if s in cat_anc and score >= min_relative_score * s_score:
                suppressed.append((s, s_score))
            else:
                keep.append((s, s_score))
        keep.append((cat, score))
        selected = keep
    remainder = list(ranked[consumed:])
    return selected + remainder + suppressed, suppressed
