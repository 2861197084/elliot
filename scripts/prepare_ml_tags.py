#!/usr/bin/env python3
import argparse
import io
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
M20_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"


def download_and_extract(url: str, extract_subdir: str) -> str:
    print(f"Downloading: {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    # Extract into a temporary folder under data/tmp
    tmp_root = os.path.join("data", "tmp_ml")
    os.makedirs(tmp_root, exist_ok=True)
    zf.extractall(tmp_root)
    path = os.path.join(tmp_root, extract_subdir)
    if not os.path.isdir(path):
        raise RuntimeError(f"Expected directory not found after extraction: {path}")
    print(f"Extracted to: {path}")
    return path


_non_alnum_re = re.compile(r"[^a-z0-9]+")


def clean_and_tokenize(text: str, min_len: int = 2) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    # normalize whitespace and remove punctuation
    text = _non_alnum_re.sub(" ", text)
    toks = [t for t in text.strip().split() if len(t) >= min_len and t not in ENGLISH_STOP_WORDS]
    return toks


def write_dataset_tsv(ratings_csv: str, out_path: str, chunksize: int = 1_000_000) -> Tuple[int, int, int]:
    cols = ["userId", "movieId", "rating", "timestamp"]
    total_rows = 0
    users = set()
    items = set()
    with open(out_path, "w") as f_out:
        for chunk in pd.read_csv(ratings_csv, usecols=cols, chunksize=chunksize):
            total_rows += len(chunk)
            users.update(chunk["userId"].unique().tolist())
            items.update(chunk["movieId"].unique().tolist())
            # write as TSV: userId\titemId\trating\ttimestamp
            chunk.to_csv(f_out, sep="\t", header=False, index=False)
    return len(users), len(items), total_rows


def build_item_features(tags_csv: str,
                        min_count: int,
                        max_per_item: int,
                        restrict_items: set = None,
                        chunksize: int = 1_000_000) -> Tuple[Dict[int, List[str]], Counter]:
    # Accumulate per-item token counts and global token frequency
    item_token_counts: Dict[int, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    cols = ["userId", "movieId", "tag", "timestamp"]
    for chunk in pd.read_csv(tags_csv, usecols=cols, chunksize=chunksize):
        if restrict_items is not None:
            chunk = chunk[chunk["movieId"].isin(restrict_items)]
        for _, row in chunk.iterrows():
            movie_id = int(row["movieId"])  # type: ignore
            toks = clean_and_tokenize(row["tag"])  # type: ignore
            if not toks:
                continue
            item_token_counts[movie_id].update(toks)
            global_counts.update(toks)

    # Filter by global min_count and limit per item
    filtered_item_tokens: Dict[int, List[str]] = {}
    for movie_id, ctr in item_token_counts.items():
        toks = [t for t, c in ctr.most_common() if global_counts[t] >= min_count]
        if max_per_item and max_per_item > 0:
            toks = toks[:max_per_item]
        if toks:
            filtered_item_tokens[movie_id] = toks

    return filtered_item_tokens, global_counts


def write_item_features_and_map(item_tokens: Dict[int, List[str]], out_features: str, out_map: str) -> Tuple[int, int]:
    # Build vocabulary sorted by decreasing global frequency then lexicographically for determinism
    # First compute global counts from provided item_tokens
    global_counts = Counter()
    for toks in item_tokens.values():
        global_counts.update(toks)
    # sort tokens by (-count, token)
    vocab = sorted(global_counts.items(), key=lambda x: (-x[1], x[0]))
    token_to_id = {tok: idx for idx, (tok, _) in enumerate(vocab)}

    # Write item_features.tsv with tag IDs
    with open(out_features, "w") as f_feat:
        for movie_id, toks in item_tokens.items():
            ids = [str(token_to_id[t]) for t in toks if t in token_to_id]
            if not ids:
                continue
            f_feat.write(str(movie_id))
            f_feat.write("\t")
            f_feat.write("\t".join(ids))
            f_feat.write("\n")

    # Write tags_map.tsv
    with open(out_map, "w") as f_map:
        for tok, idx in sorted(token_to_id.items(), key=lambda x: x[1]):
            f_map.write(f"{idx}\t{tok}\n")

    return len(token_to_id), sum(len(v) for v in item_tokens.values())


def main():
    parser = argparse.ArgumentParser(description="Prepare MovieLens ratings and tag features for Elliot.")
    parser.add_argument("--dataset", choices=["small", "20m"], default="20m", help="Choose MovieLens variant.")
    parser.add_argument("--min-count", type=int, default=3, help="Global min frequency for a tag token to be kept.")
    parser.add_argument("--max-tags-per-item", type=int, default=50, help="Max tags per item after filtering (0 = no limit).")
    parser.add_argument("--out-root", type=str, default="data", help="Root data folder.")

    args = parser.parse_args()

    if args.dataset == "small":
        url = SMALL_URL
        subdir = "ml-latest-small"
        out_dir_name = "movielens_small"
    else:
        url = M20_URL
        subdir = "ml-20m"
        out_dir_name = "movielens_20m"

    src_dir = download_and_extract(url, subdir)

    out_dir = os.path.join(args.out_root, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    ratings_csv = os.path.join(src_dir, "ratings.csv")
    tags_csv = os.path.join(src_dir, "tags.csv")

    if not os.path.isfile(ratings_csv):
        print(f"ratings.csv not found at {ratings_csv}")
        sys.exit(1)
    if not os.path.isfile(tags_csv):
        print(f"tags.csv not found at {tags_csv} (this dataset variant may not have tags)")
        sys.exit(1)

    dataset_tsv = os.path.join(out_dir, "dataset.tsv")
    item_features_tsv = os.path.join(out_dir, "item_features.tsv")
    tags_map_tsv = os.path.join(out_dir, "tags_map.tsv")

    print("Writing dataset.tsv ...")
    n_users, n_items, n_rows = write_dataset_tsv(ratings_csv, dataset_tsv)
    print(f"Ratings written: users={n_users}, items={n_items}, rows={n_rows}")

    print("Building tag features ...")
    # restrict item set to those present in ratings
    # Read back unique item IDs from the dataset we just wrote to avoid re-parsing CSV into memory
    # Lightweight pass using pandas
    df_items = pd.read_csv(dataset_tsv, sep="\t", header=None, names=["userId", "itemId", "rating", "timestamp"], usecols=[1])
    restrict_items = set(df_items["itemId"].unique().tolist())

    item_tokens, global_counts = build_item_features(tags_csv,
                                                    min_count=args.min_count,
                                                    max_per_item=args.max_tags_per_item,
                                                    restrict_items=restrict_items)
    kept_items = len(item_tokens)
    print(f"Items with tags after filtering: {kept_items} / {len(restrict_items)}")

    print("Writing item_features.tsv and tags_map.tsv ...")
    n_tags, total_assignments = write_item_features_and_map(item_tokens, item_features_tsv, tags_map_tsv)
    print(f"Tags in vocab: {n_tags}, total assignments: {total_assignments}")

    print("Done. You can now point Elliot config to:")
    print(f"  dataset_path: ../{out_dir}/dataset.tsv")
    print(f"  attribute_file: ../{out_dir}/item_features.tsv")


if __name__ == "__main__":
    main()

