"""Fetch the independently selected SGD evaluation shards, pinned to the pilot's revision."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from scripts.fetch_sgd import get_file, REVISION, REPO


FRESH_SHARDS = (2, 3, 15, 17, 29, 31, 43, 45, 57, 59, 71, 73, 85, 87, 99, 101, 113, 115, 127)
REPOSITORY = "google-research-datasets/dstc8-schema-guided-dialogue"
LOCK = REPO / "configs" / "support_holdout_sha256.json"


def fetch(out):
    names = ["schema.json", *(f"dialogues_{n:03}.json" for n in FRESH_SHARDS)]
    target = out / "train"
    target.mkdir(parents=True, exist_ok=True)
    if {f.name for f in target.glob("dialogues_*.json")} - set(names):
        raise ValueError("Output directory contains other SGD shards")
    expected = json.loads(LOCK.read_bytes())
    if set(expected) != set(names):
        raise ValueError("Fresh shard lock does not match the configured shard list")
    with ThreadPoolExecutor(max_workers=5) as pool:
        hashes = dict(pool.map(lambda name: get_file(name, target, expected[name]), names))
    manifest = {"repository": REPOSITORY, "revision": REVISION, "split": "train",
                "profile": "support-holdout", "sha256": hashes}
    data = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    path = out / "manifest.json"
    path.write_bytes(data)
    print(f"{len(FRESH_SHARDS)} validation shards, manifest SHA-256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "sgd-support-holdout")
    fetch(parser.parse_args().out)
