# Ground-truth labels for the F18-F20 validation split (data/testingLabel.xlsx).
#
# The sheet stores one row per aircraft with two range strings, e.g.
#   F19 | Faulty "622-662(15)" | Healthy "663-689(17)"
# The bracketed count is the number of flights actually present, which is
# smaller than the span whenever sorties are missing -- so a flight is labelled
# by which range its sortie number falls into, not by position.
import re

import pandas as pd

from ..config import TESTING_LABEL_XLSX

RANGE_RE = re.compile(r"(?P<lo>\d+)\s*-\s*(?P<hi>\d+)\s*\((?P<n>\d+)\)")

LABEL_HEALTHY = 0
LABEL_FAULTY = 1


def load_testing_ranges(xlsx_path=TESTING_LABEL_XLSX) -> pd.DataFrame:
    """-> DataFrame of (plane, label, lo, hi, n_expected), one row per range."""
    raw = pd.read_excel(xlsx_path)
    plane_col = raw.columns[0]

    rows = []
    for _, r in raw.iterrows():
        plane = str(r[plane_col]).strip()
        for col, label in (("Faulty", LABEL_FAULTY), ("Healthy", LABEL_HEALTHY)):
            m = RANGE_RE.search(str(r[col]))
            if not m:
                raise ValueError(f"{xlsx_path}: cannot parse {plane} {col}: {r[col]!r}")
            rows.append({
                "plane": plane, "label": label,
                "lo": int(m["lo"]), "hi": int(m["hi"]), "n_expected": int(m["n"]),
            })
    return pd.DataFrame(rows)


def label_testing_flights(planes, flights, xlsx_path=TESTING_LABEL_XLSX) -> pd.Series:
    """Map (plane, flight) pairs to 0/1 labels; raises if any flight is unlabelled."""
    ranges = load_testing_ranges(xlsx_path)
    idx = pd.MultiIndex.from_arrays([list(planes), list(flights)], names=["plane", "flight"])
    out = pd.Series(pd.NA, index=idx, dtype="Int64")

    for _, r in ranges.iterrows():
        hit = [
            (p, f) for p, f in idx
            if p == r["plane"] and r["lo"] <= f <= r["hi"]
        ]
        # The sheet counts sorties that were labelled, not sorties that shipped
        # with the dataset; a few (F18-00579, one F19 healthy flight) have no CSV.
        # Missing files are fine — extra unlabelled ones are not, and are caught below.
        if len(hit) != r["n_expected"]:
            print(f"[labels][WARN] {r['plane']} {r['lo']}-{r['hi']}: sheet lists "
                  f"{r['n_expected']} flights, {len(hit)} present on disk")
        for key in hit:
            if not pd.isna(out.loc[key]):
                raise ValueError(f"{key} matches more than one label range")
            out.loc[key] = r["label"]

    missing = out[out.isna()]
    if len(missing):
        raise ValueError(f"{len(missing)} testing flights fall outside every label range: "
                         f"{list(missing.index)[:10]}")
    return out.astype(int)
