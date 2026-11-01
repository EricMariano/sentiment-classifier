"""O tradutor muda a satisfação que o Jev estima? Mesmas conversas, duas traduções.

Para cada conversa presente nos dois arquivos: anonimiza, pergunta ao Jev a satisfação (uma
requisição, com cache) e compara as duas versões entre si e com a nota real.

    python benchmarks/uss/translator_check.py benchmarks/data/jddc_90_pt.jsonl benchmarks/data/jddc_90_zh_pt.jsonl
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from conversa_lens.anonymize import redact_turns
from conversa_lens.csat import _spearman
from conversa_lens.parsing import turns_from_messages
from conversa_lens.providers import make_csat_estimator


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    a_path, b_path = map(Path, sys.argv[1:3])
    a, b = load(a_path), load(b_path)
    ids = sorted(set(a) & set(b))
    estimator = make_csat_estimator()
    assert estimator is not None, "precisa de TYPESAFE_API_KEY"

    def csat(row: dict) -> float:
        turns = redact_turns(turns_from_messages(row["messages"])).turns
        return estimator.estimate(turns).value

    with ThreadPoolExecutor(8) as pool:
        ea = np.array(list(pool.map(lambda i: csat(a[i]), ids)))
        eb = np.array(list(pool.map(lambda i: csat(b[i]), ids)))
    y = np.array([a[i]["nota"] for i in ids])
    name = lambda r: r.get("tradutor", "?").split(":")[-1]  # noqa: E731
    na, nb = name(a[ids[0]]), name(b[ids[0]])

    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(ids), len(ids))
        diffs.append(_spearman(eb[i], y[i]) - _spearman(ea[i], y[i]))

    print(f"{len(ids)} conversas com as duas traduções\n")
    print(
        f"mesma conversa, tradutores diferentes: Spearman {_spearman(ea, eb):.2f}, "
        f"diferença média {np.mean(eb - ea):+.2f}, |diferença| média {np.mean(abs(eb - ea)):.2f}"
    )
    for n, e in ((na, ea), (nb, eb)):
        bad, flag = y < 2.5, e <= 2.5
        print(
            f"{n:<36} vs nota real: Spearman {_spearman(e, y):.2f}  MAE {np.mean(abs(e - y)):.2f}  "
            f"insatisfeitas achadas {int((bad & flag).sum())}/{int(bad.sum())}"
        )
    print(
        f"\nΔSpearman ({nb} − {na}): {np.mean(diffs):+.2f}  "
        f"IC95% [{np.percentile(diffs, 2.5):+.2f}, {np.percentile(diffs, 97.5):+.2f}]"
    )


if __name__ == "__main__":
    main()
