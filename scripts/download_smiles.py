#!/usr/bin/env python
"""Download a real SMILES dataset for small-molecule benchmarking.

Default source is the classic 250k random drug-like subset of ZINC (used by
many generative-chemistry papers). Writes one canonical SMILES per line, ready
for `denovo prepare` / `denovo evaluate` (as the novelty reference set).

    python scripts/download_smiles.py --max 50000 -o data/zinc.txt

No extra dependencies (uses urllib). Runs on your machine where the network is
open; it will not work in a sandbox that blocks outbound HTTP.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request

# 250k random drug-like ZINC molecules (single 'smiles' column CSV).
ZINC_250K = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc/250k_rndm_zinc_drugs_clean_3.csv"
)

SOURCES = {"zinc": ZINC_250K}


def download(url: str, out: str, max_n: int) -> int:
    print(f"GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "De-Novo-LLM/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = io.TextIOWrapper(resp, encoding="utf-8")
        reader = csv.reader(text)
        written = 0
        with open(out, "w", encoding="utf-8") as fh:
            for i, row in enumerate(reader):
                if not row:
                    continue
                cell = row[0].strip().strip('"')
                if i == 0 and cell.lower() in ("smiles", "smile"):
                    continue  # header
                smi = cell.split()[0] if cell else ""
                if smi:
                    fh.write(smi + "\n")
                    written += 1
                if written >= max_n:
                    break
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=list(SOURCES), default="zinc")
    ap.add_argument("--url", help="Override with any CSV/txt URL whose first column is SMILES.")
    ap.add_argument("--out", "-o", default="data/zinc.txt")
    ap.add_argument("--max", type=int, default=50000, help="Max molecules to keep.")
    args = ap.parse_args()

    url = args.url or SOURCES[args.source]
    try:
        n = download(url, args.out, args.max)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {n} SMILES to {args.out}")
    print(f"Next:  denovo prepare -c configs/molecule_benchmark.yaml -i {args.out}")


if __name__ == "__main__":
    main()
