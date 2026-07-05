"""NVIDIA NIM (BioNeMo) cloud-inference client.

Calls NVIDIA's hosted biomolecule NIMs so you can use models that don't fit a
local GPU — **MolMIM** (controlled small-molecule generation / property
optimization), **ESMFold** (sequence → 3D structure), and **Evo 2** (genomic
generation) — via a simple API. This complements the local tracks: generate or
fine-tune locally, then call a NIM for capabilities that need cloud scale.

Auth: set an API key from https://build.nvidia.com (free NVIDIA Developer
Program) in the ``NVIDIA_API_KEY`` environment variable::

    # Windows CMD:   set NVIDIA_API_KEY=nvapi-xxxx
    # bash:          export NVIDIA_API_KEY=nvapi-xxxx

No extra dependencies (uses urllib). Async NIM responses (HTTP 202) are polled
automatically.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

HEALTH_BASE = "https://health.api.nvidia.com/v1/biology"
STATUS_URL = "https://health.api.nvidia.com/v1/status/"

#: Hosted NIMs this client wraps, with their catalog pages.
NIM_MODELS = {
    "molmim": "nvidia/molmim — controlled small-molecule generation & optimization",
    "esmfold": "nvidia/esmfold — protein sequence → 3D structure",
    "evo2": "arc/evo2-40b — genomic (DNA) sequence generation",
}


class NIMClient:
    """Thin client for NVIDIA hosted biology NIMs."""

    def __init__(self, api_key: Optional[str] = None, base: str = HEALTH_BASE,
                 poll_interval: float = 2.0, timeout: int = 300):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No API key. Get one at https://build.nvidia.com and set "
                "NVIDIA_API_KEY (e.g. `set NVIDIA_API_KEY=nvapi-...`)."
            )
        self.base = base
        self.poll_interval = poll_interval
        self.timeout = timeout

    # -- HTTP plumbing ---------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _post(self, url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"NIM request failed [{exc.status}]: {body}") from exc
        return self._resolve(resp)

    def _resolve(self, resp) -> dict:
        """Return the JSON body, polling the status endpoint on async (202)."""
        if getattr(resp, "status", resp.getcode()) == 202:
            reqid = resp.headers.get("NVCF-REQID") or resp.headers.get("nvcf-reqid")
            if not reqid:
                raise RuntimeError("Async NIM response (202) without a request id.")
            deadline = self.timeout
            waited = 0.0
            while True:
                poll = urllib.request.Request(STATUS_URL + reqid, headers=self._headers())
                r = urllib.request.urlopen(poll, timeout=self.timeout)
                if getattr(r, "status", r.getcode()) != 202:
                    return json.loads(r.read().decode("utf-8"))
                time.sleep(self.poll_interval)
                waited += self.poll_interval
                if waited > deadline:
                    raise RuntimeError("Timed out polling NIM async result.")
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}

    # -- MolMIM: small-molecule generation / optimization ----------------
    def molmim_generate(
        self,
        smi: str,
        *,
        num_molecules: int = 30,
        algorithm: str = "CMA-ES",
        property_name: str = "QED",
        minimize: bool = False,
        min_similarity: float = 0.3,
        particles: int = 30,
        iterations: int = 10,
        scaled_radius: float = 1.0,
    ) -> List[Dict]:
        """Generate molecules around a seed SMILES, optionally optimizing a property.

        ``algorithm="none"`` samples the latent space; ``"CMA-ES"`` optimizes
        ``property_name`` (e.g. ``QED``, ``plogP``). Returns dicts with
        ``smiles`` and, when optimizing, ``score`` / ``similarity``.
        """
        payload = {
            "smi": smi,
            "num_molecules": num_molecules,
            "algorithm": algorithm,
            "property_name": property_name,
            "minimize": minimize,
            "min_similarity": min_similarity,
            "particles": particles,
            "iterations": iterations,
            "scaled_radius": scaled_radius,
        }
        out = self._post(f"{self.base}/nvidia/molmim/generate", payload)
        mols = out.get("molecules", out.get("generated", []))
        if isinstance(mols, str):
            mols = json.loads(mols)
        result = []
        for m in mols:
            if isinstance(m, str):
                result.append({"smiles": m})
            else:
                result.append({
                    "smiles": m.get("sample") or m.get("smiles"),
                    "score": m.get("score"),
                    "similarity": m.get("similarity"),
                })
        return result

    # -- ESMFold: sequence -> structure ----------------------------------
    def esmfold_predict(self, sequence: str) -> Optional[str]:
        """Fold a protein sequence; returns the predicted structure as PDB text."""
        out = self._post(f"{self.base}/nvidia/esmfold", {"sequence": sequence})
        if "pdbs" in out and out["pdbs"]:
            return out["pdbs"][0]
        return out.get("output_pdb") or out.get("pdb")

    # -- Evo 2: genomic generation ---------------------------------------
    def evo2_generate(self, sequence: str, *, num_tokens: int = 100,
                      temperature: float = 1.0, top_k: int = 4) -> Optional[str]:
        """Continue a DNA sequence with Evo 2 (returns the generated string)."""
        payload = {"sequence": sequence, "num_tokens": num_tokens,
                   "temperature": temperature, "top_k": top_k}
        out = self._post(f"{self.base}/arc/evo2-40b/generate", payload)
        return out.get("generated_sequence") or out.get("sequence") or out.get("output")
