"""
One place that decides which rows of the corpus tables a headline number is computed on.

P1-4 appends external-validity cells (p = 97, 113, sparse parity) to operating_point.csv and
e1_centroid.csv, tagged source = "P1-4". Every existing headline number in the paper is defined on
the original corpus, so consumers must not silently absorb the new rows -- a merge that moved
Table 1 underneath the text would be exactly the kind of failure the validity gates exist to catch.
Read through here instead of pd.read_csv.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1] / "RESULTS"


def read_corpus(name: str, source: str | None = "corpus") -> pd.DataFrame:
    """source="corpus" -> the original rows only (the default, and what the paper's numbers use);
    source=None -> everything, including the P1-4 external-validity cells."""
    d = pd.read_csv(R / name)
    if source is None or "source" not in d.columns:
        return d
    return d[d["source"].fillna("corpus") == source].reset_index(drop=True)
