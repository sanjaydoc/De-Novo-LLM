#!/usr/bin/env python
"""Download real training data for fine-tuning (no extra dependencies).

Proteins are pulled from UniProt's REST API. By default it fetches reviewed
(Swiss-Prot) sequences within a length range and writes one sequence per line,
ready for ``denovo prepare``.

Examples
--------
    # 20k reviewed human proteins, 40-256 residues
    python scripts/download_data.py --query 'reviewed:true AND organism_id:9606' \
        --min-len 40 --max-len 256 --max 20000 --out data/uniprot_human.txt

    # A protein family (e.g. PDZ, Pfam PF00595)
    python scripts/download_data.py --query 'xref:pfam-PF00595' \
        --max 5000 --out data/pdz.txt
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
import urllib.parse
import urllib.request

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"


def fetch_uniprot(query: str, out: str, min_len: int, max_len: int, max_n: int) -> int:
    params = {"query": query, "format": "fasta", "compressed": "true"}
    url = UNIPROT_STREAM + "?" + urllib.parse.urlencode(params)
    print(f"GET {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "De-Novo-LLM/0.1"})
    written = 0
    seq_parts: list[str] = []

    def flush(parts, fh):
        nonlocal written
        if not parts:
            return
        seq = "".join(parts)
        if min_len <= len(seq) <= max_len:
            fh.write(seq + "\n")
            written += 1

    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = gzip.GzipFile(fileobj=io.BytesIO(resp.read()))
        text = io.TextIOWrapper(raw, encoding="utf-8")
        with open(out, "w", encoding="utf-8") as fh:
            for line in text:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    flush(seq_parts, fh)
                    seq_parts = []
                    if written >= max_n:
                        break
                else:
                    seq_parts.append(line.strip())
            else:
                flush(seq_parts, fh)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", default="reviewed:true",
                    help="UniProt query string (default: all Swiss-Prot).")
    ap.add_argument("--out", "-o", default="data/uniprot.txt")
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max", type=int, default=50000, help="Max sequences to keep.")
    args = ap.parse_args()

    try:
        n = fetch_uniprot(args.query, args.out, args.min_len, args.max_len, args.max)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Download failed: {exc}", file=sys.stderr)
        print("Check your network / query. UniProt syntax: https://www.uniprot.org/help/query-fields",
              file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {n} sequences to {args.out}")
    print(f"Next:  denovo prepare -c configs/progen2_protein.yaml -i {args.out}")


if __name__ == "__main__":
    main()
