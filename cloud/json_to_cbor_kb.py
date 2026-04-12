import json
from pathlib import Path

import cbor2


src = Path("results/models/retrieval_kb.json")
dst = Path("results/models/retrieval_kb.cbor")

with src.open("r", encoding="utf-8") as f:
    kb = json.load(f)

with dst.open("wb") as f:
    cbor2.dump(kb, f)

print(f"Wrote {dst}")
