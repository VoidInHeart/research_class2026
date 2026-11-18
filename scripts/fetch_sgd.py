"""Download the public SGD train split at one immutable Git revision. No API key."""
import argparse
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

REVISION = "e852981ae34990f4358979625854259302feaa78"
BASE = f"https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/{REVISION}/train"
PILOT = (1, 14, 28, 42, 56, 70, 84, 98, 112, 126)
REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "configs" / "sgd_pilot_sha256.json"


def get_file(name, dest, expected):
    target = dest / name
    if target.exists() and expected and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        return name, expected
    with tempfile.NamedTemporaryFile(dir=dest, suffix=".part", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        for attempt in range(3):
            try:
                with urlopen(f"{BASE}/{name}", timeout=60) as response, temporary.open("wb") as out:
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
                break
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raw = temporary.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected and expected != digest:
            raise ValueError(f"Bad digest for {name}: {digest}")
        if not isinstance(json.loads(raw), list):
            raise ValueError(f"Invalid SGD JSON for {name}")
        os.replace(temporary, target)
        return name, digest
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "sgd")
    parser.add_argument("--all", action="store_true", help="Fetch all 127 training shards into a separate dataset directory")
    args = parser.parse_args()
    dest = args.out / "train"
    dest.mkdir(exist_ok=True, parents=True)
    shards = range(1, 128) if args.all else PILOT
    names = ["schema.json"] + [f"dialogues_{i:03}.json" for i in shards]
    extra = {p.name for p in dest.glob("dialogues_*.json")} - set(names)
    if extra:
        parser.error("Directory contains shards outside the requested profile. Use a separate --out directory.")
    expected = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        digests = dict(pool.map(lambda name: get_file(name, dest, expected.get(name)), names))
    manifest = {"repository": "google-research-datasets/dstc8-schema-guided-dialogue", "revision": REVISION,
                "split": "train", "profile": "all" if args.all else "pilot", "sha256": digests}
    with tempfile.NamedTemporaryFile(dir=args.out, mode="w", encoding="utf-8", delete=False) as tmp:
        json.dump(manifest, tmp, indent=2)
        tmp.write("\n")
        manifest_temp = Path(tmp.name)
    os.replace(manifest_temp, args.out / "manifest.json")
    print(f"Ready: {len(names)-1} shards, revision {REVISION}, metadata {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
