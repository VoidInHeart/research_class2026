"""Fetch the fixed ten new SGD shards used for routing confirmation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from scripts.fetch_sgd import get_file, REPO, REVISION
from scripts.fetch_support_holdout import REPOSITORY


SHARDS = (5, 18, 32, 46, 60, 74, 88, 102, 116, 122)
LOCK = REPO / "configs" / "refresh_holdout_sha256.json"


def fetch(out):
    names = ["schema.json", *(f"dialogues_{n:03}.json" for n in SHARDS)]
    directory = out / "train"
    directory.mkdir(parents=True, exist_ok=True)
    if {p.name for p in directory.glob("dialogues_*.json")} - set(names):
        raise ValueError("Fresh input directory contains unselected shards")
    expected = json.loads(LOCK.read_bytes())
    if set(expected) != set(names):
        raise ValueError("Invalid fresh shard list")
    with ThreadPoolExecutor(max_workers=5) as workers:
        hashes = dict(workers.map(lambda n: get_file(n, directory, expected[n]), names))
    manifest = {"repository": REPOSITORY, "revision": REVISION, "split": "train",
                "profile": "routing-confirmation", "sha256": hashes}
    (out / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "sgd-routing-confirmation")
    fetch(parser.parse_args().out)
