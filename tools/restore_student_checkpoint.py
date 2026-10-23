"""Restore the complete front Student + optimizer checkpoint from verified GitHub parts."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def restore(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    target = directory / manifest["checkpoint"]
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != manifest["sha256"]:
            raise FileExistsError(f"Preserving different existing checkpoint: {target}")
        return target
    parts = directory / manifest["storage"]["parts_dir"]
    # Temporary files live on the destination filesystem for atomic final rename.
    with tempfile.TemporaryDirectory(prefix="restore-", dir=directory) as temporary:
        archive = Path(temporary) / "checkpoint.gz"
        digest = hashlib.sha256()
        with archive.open("wb") as output:
            for entry in manifest["storage"]["parts"]:
                path = parts / entry["file"]
                content = path.read_bytes()
                if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Checkpoint part integrity failure: {path}")
                digest.update(content)
                output.write(content)
        if digest.hexdigest() != manifest["storage"]["compressed_sha256"]:
            raise ValueError("Compressed checkpoint hash mismatch")
        decoded = Path(temporary) / manifest["checkpoint"]
        with gzip.open(archive, "rb") as source, decoded.open("wb") as output:
            shutil.copyfileobj(source, output)
        if decoded.stat().st_size != manifest["bytes"] or hashlib.sha256(decoded.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("Restored checkpoint hash mismatch")
        if target.exists():
            raise FileExistsError(f"Destination appeared during restore: {target}")
        os.replace(decoded, target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / "checkpoints/rbq10_student_front")
    args = parser.parse_args()
    print(restore(args.directory.resolve()))
