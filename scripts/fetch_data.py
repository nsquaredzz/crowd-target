#!/usr/bin/env python3
"""Download MOT20 into data/ and verify it.

    python scripts/fetch_data.py            # asks before the 5.03 GB download
    python scripts/fetch_data.py --verify   # re-hash what is on disk

MOT20 (Dendorfer et al., 2020) is hosted by the MOTChallenge at motchallenge.net and stays under
its terms (CC BY-NC-SA 3.0); nothing is redistributed here.  Standard library only; written to a
.part name first and renamed once size and SHA-256 match.  The digest was recorded on the first
download (2026-10-15) and is verified ever after.
"""
import argparse
import hashlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FILES = [
    ("mot20/MOT20.zip", "https://motchallenge.net/data/MOT20.zip", 5028926248,
     "ebcf0e3d44e4f50b5357d24817e5db485d777633d1b8ca9e8380d1c8437dbdd7"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def good(path, size, digest):
    return path.exists() and path.stat().st_size == size and (digest is None or sha256(path) == digest)


def fetch(rel, url, size, digest):
    path = DATA / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if good(path, size, digest):
        print(f"ok       {rel}")
        return True
    if input(f"{rel} is {size / 1e9:.2f} GB from {url}. Download? [y/N] ").strip().lower() != "y":
        return False
    part = path.with_suffix(path.suffix + ".part")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "crowd-target-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        done = 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done % (1 << 28) < (1 << 22):
                print(f"  {done / 1e9:.2f} GB", flush=True)
    if part.stat().st_size != size:
        print(f"size mismatch for {rel}", file=sys.stderr)
        return False
    d = sha256(part)
    if digest is None:
        print(f"first download of {rel}: sha256 {d}; paste it into FILES")
    elif d != digest:
        print(f"sha256 mismatch for {rel}", file=sys.stderr)
        return False
    part.rename(path)
    print(f"done     {rel} in {time.time() - t0:.0f} s")
    return True


def unzip(rel):
    path = DATA / rel
    with zipfile.ZipFile(path) as z:
        top = path.parent / z.namelist()[0].split("/")[0]
        if top.exists():
            return
        z.extractall(path.parent)
    print(f"unzipped {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    ok = True
    for rel, url, size, digest in FILES:
        if a.verify:
            print(("ok       " if good(DATA / rel, size, digest) else "BAD      ") + rel)
            continue
        if fetch(rel, url, size, digest):
            unzip(rel)
        else:
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
