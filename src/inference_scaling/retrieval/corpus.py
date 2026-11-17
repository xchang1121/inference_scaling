"""Deterministic local retrieval, with no questions or answer labels in the index."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re


def terms(text: str) -> list[str]:
    # Keep dotted versions and API identifiers, also expose their constituent words.
    whole = re.findall(r"[\w]+(?:[./:-][\w]+)*", text.casefold())
    return whole + [piece for word in whole if re.search(r"[_.:/-]", word)
                    for piece in re.split(r"[_.:/-]+", word) if piece != word]


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class Hit:
    document: Document
    score: float


class Retriever:
    """BM25, character TF-IDF, or reciprocal-rank fusion of the two.

    Character matching handles error codes/version strings; it is not a semantic
    embedding retriever. Index construction is deterministic and dependency free.
    """

    def __init__(self, documents: list[Document], *, kind: str = "hybrid") -> None:
        if kind not in {"bm25", "char", "hybrid"}:
            raise ValueError("retriever kind must be bm25, char, or hybrid")
        if not documents or len({d.id for d in documents}) != len(documents):
            raise ValueError("corpus must be nonempty with unique document IDs")
        self.documents = tuple(sorted(documents, key=lambda d: d.id))
        self.kind = kind
        self._built = frozenset({"bm25", "char"} if kind == "hybrid" else {kind})
        self.fingerprint = hashlib.sha256(json.dumps(
            [d.__dict__ for d in self.documents], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.lengths = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.char_postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        chars = []
        for index, doc in enumerate(self.documents):
            counts = Counter(terms(doc.title + " " + doc.title + " " + doc.text)) if "bm25" in self._built else Counter()
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((index, count))
            if "char" in self._built:
                compact = re.sub(r"\s+", " ", (doc.title + " " + doc.text).casefold())
                chars.append(Counter(compact[i:i+3] for i in range(len(compact)-2)))
        self.average_length = sum(self.lengths) / len(documents) or 1
        df = Counter(c for counts in chars for c in counts)
        self.char_idf = {c: math.log((1+len(documents))/(1+count)) + 1 for c, count in df.items()}
        for index, counts in enumerate(chars):
            weights = {c: (1+math.log(n))*self.char_idf[c] for c, n in counts.items()}
            norm = math.sqrt(sum(w*w for w in weights.values())) or 1
            for c, weight in weights.items():
                self.char_postings[c].append((index, weight/norm))

    def with_kind(self, kind: str) -> Retriever:
        required = {"bm25", "char"} if kind == "hybrid" else {kind}
        if not required <= self._built:
            raise ValueError("requested index was not built")
        view = object.__new__(Retriever)
        view.__dict__ = {**self.__dict__, "kind": kind}
        return view

    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs) -> Retriever:
        documents = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if set(row) - {"id", "title", "text", "source"}:
                raise ValueError("corpus accepts document fields only; strip QA annotations first")
            documents.append(Document(**row))
        return cls(documents, **kwargs)

    def _bm25(self, query: str) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        n = len(self.documents)
        for term, frequency in Counter(terms(query)).items():
            posting = self.postings.get(term, [])
            idf = math.log(1 + (n - len(posting) + .5)/(len(posting) + .5))
            for index, tf in posting:
                scores[index] += min(frequency, 3) * idf * tf * 2.5 / (
                    tf + 1.5 * (.25 + .75 * self.lengths[index]/self.average_length))
        return dict(scores)

    def _char(self, query: str) -> dict[int, float]:
        compact = re.sub(r"\s+", " ", query.casefold())
        counts = Counter(compact[i:i+3] for i in range(len(compact)-2))
        weights = {c: (1+math.log(n))*self.char_idf[c] for c, n in counts.items() if c in self.char_idf}
        norm = math.sqrt(sum(w*w for w in weights.values())) or 1
        scores: dict[int, float] = defaultdict(float)
        for c, weight in weights.items():
            for index, document_weight in self.char_postings[c]:
                scores[index] += weight/norm * document_weight
        return dict(scores)

    def search(self, query: str, *, k: int = 3, exclude: frozenset[str] = frozenset()) -> list[Hit]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not query.strip():
            return []
        def rank(scores):
            return sorted((i for i, score in scores.items() if score > 0 and self.documents[i].id not in exclude),
                          key=lambda i: (-scores[i], self.documents[i].id))
        if self.kind == "hybrid":
            scores: dict[int, float] = defaultdict(float)
            for base in (self._bm25(query), self._char(query)):
                for position, index in enumerate(rank(base)[:100], 1):
                    scores[index] += 1/(60+position)
        else:
            scores = self._bm25(query) if self.kind == "bm25" else self._char(query)
        return [Hit(self.documents[i], scores[i]) for i in rank(scores)[:k]]
