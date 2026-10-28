"""Reproducible split manifests with duplicate/near-duplicate leakage prevention.

Images are grouped by exact hash and perceptual dHash proximity. Whole groups are assigned to
one split, so near-identical images can never straddle train/val/test.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from shelfsight.data.formats import IMAGE_EXTENSIONS, DatasetError

logger = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")
MANIFEST_COLUMNS = ["image", "split", "sha256", "dhash", "group_id", "source", "license"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_image(path: Path, hash_size: int = 8) -> int:
    """64-bit difference hash (row gradients of a 9x8 grayscale thumbnail)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise DatasetError(f"Cannot decode image for hashing: {path}")
    small = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_matrix(hashes: list[int]) -> np.ndarray:
    arr = np.array(hashes, dtype=np.uint64)
    return np.bitwise_count(arr[:, None] ^ arr[None, :])


def group_images(hashes: list[int], threshold: int) -> list[int]:
    """Union-find over pairs with Hamming distance <= threshold. Returns group id per image."""
    n = len(hashes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n:
        close = np.argwhere(np.triu(hamming_matrix(hashes) <= threshold, k=1))
        for i, j in close:
            parent[find(int(i))] = find(int(j))
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [remap[r] for r in roots]


def build_manifest(
    image_dir: str | Path,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    near_dup_threshold: int = 4,
    sources: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a deterministic group-aware split manifest for the images in ``image_dir``.

    ``sources`` optionally has columns image, source, license; images without an entry are
    recorded as ``unknown`` (``verify_splits`` flags these as unverified usage permission).
    """
    if abs(sum(ratios) - 1.0) > 1e-6 or any(r <= 0 for r in ratios):
        raise ValueError("ratios must be positive and sum to 1")
    root = Path(image_dir)
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise DatasetError(f"No images found in {root}")
    sha = [sha256_file(p) for p in files]
    dh = [dhash_image(p) for p in files]
    groups = group_images(dh, near_dup_threshold)
    n_groups = len(set(groups))
    if n_groups < len(SPLITS):
        raise DatasetError(f"Need at least {len(SPLITS)} distinct image groups, found {n_groups}")

    members: dict[int, list[int]] = {}
    for idx, g in enumerate(groups):
        members.setdefault(g, []).append(idx)
    order = sorted(members)
    np.random.default_rng(seed).shuffle(order)
    targets = np.array(ratios) * len(files)
    assigned = np.zeros(len(SPLITS))
    split_of_group: dict[int, str] = {}
    # Seed each split with one group so none is empty, then fill by largest deficit.
    for k, g in enumerate(order[: len(SPLITS)]):
        split_of_group[g] = SPLITS[k]
        assigned[k] += len(members[g])
    for g in order[len(SPLITS) :]:
        k = int(np.argmax(targets - assigned))
        split_of_group[g] = SPLITS[k]
        assigned[k] += len(members[g])

    manifest = pd.DataFrame(
        {
            "image": [p.name for p in files],
            "split": [split_of_group[g] for g in groups],
            "sha256": sha,
            "dhash": [f"{h:016x}" for h in dh],
            "group_id": groups,
        }
    )
    if sources is not None:
        manifest = manifest.merge(sources[["image", "source", "license"]], on="image", how="left")
    else:
        manifest["source"] = None
        manifest["license"] = None
    manifest["source"] = manifest["source"].fillna("unknown")
    manifest["license"] = manifest["license"].fillna("unknown")
    return manifest[MANIFEST_COLUMNS]


def verify_splits(
    manifest: pd.DataFrame, image_dir: str | Path | None = None, near_dup_threshold: int = 4
) -> list[str]:
    """Return a list of problems (empty list = no leakage or integrity problems found).

    Entries starting with ``WARNING`` are non-fatal (e.g. unknown image provenance).
    """
    problems: list[str] = []
    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        return [f"manifest missing columns: {missing}"]
    bad = sorted(set(manifest["split"]) - set(SPLITS))
    if bad:
        problems.append(f"unknown split names: {bad}")
    if manifest["image"].duplicated().any():
        problems.append("duplicate image names in manifest")
    for _sha, g in manifest.groupby("sha256"):
        if g["split"].nunique() > 1:
            problems.append(
                f"exact duplicate across splits {sorted(g['split'].unique())}: {list(g['image'])}"
            )
    hashes = [int(h, 16) for h in manifest["dhash"]]
    splits = manifest["split"].to_numpy()
    close = np.argwhere(np.triu(hamming_matrix(hashes) <= near_dup_threshold, k=1))
    for i, j in close:
        if splits[i] != splits[j]:
            problems.append(
                f"near-duplicate across splits: {manifest['image'].iloc[i]} ({splits[i]}) "
                f"~ {manifest['image'].iloc[j]} ({splits[j]})"
            )
    if image_dir is not None:
        root = Path(image_dir)
        for row in manifest.itertuples(index=False):
            p = root / row.image
            if not p.is_file():
                problems.append(f"manifest image missing on disk: {row.image}")
            elif sha256_file(p) != row.sha256:
                problems.append(f"file changed since manifest was built: {row.image}")
    n_unknown = int(
        (manifest["source"] == "unknown").sum() + (manifest["license"] == "unknown").sum()
    )
    if n_unknown:
        problems.append(
            f"WARNING: {n_unknown} source/license fields are 'unknown' (permissions unverified)"
        )
    return problems
