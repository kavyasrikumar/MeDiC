import os
import json
import random

INPUT_DIR = "data/processed/MM_docs"   # folder containing 4000+ JSON files
OUT_DIR = "data/doc_splits"
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

os.makedirs(OUT_DIR, exist_ok=True)

def load_all_json_files(directory):
    """Load all .json files from a directory into a single list."""
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            full_path = os.path.join(directory, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                try:
                    doc = json.load(f)
                    docs.append(doc)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON file: {filename}")
    return docs


def split_docs(docs, train_ratio, dev_ratio, test_ratio, seed=42):
    random.seed(seed)
    random.shuffle(docs)

    n = len(docs)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    train_docs = docs[:train_end]
    dev_docs = docs[train_end:dev_end]
    test_docs = docs[dev_end:]

    return train_docs, dev_docs, test_docs


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    print(f"Loading JSON files from: {INPUT_DIR}")
    docs = load_all_json_files(INPUT_DIR)
    print(f"Loaded {len(docs)} documents.\n")

    print("Splitting into train/dev/test…")
    train_docs, dev_docs, test_docs = split_docs(
        docs, TRAIN_RATIO, DEV_RATIO, TEST_RATIO
    )

    print("Saving files…")
    save_json(os.path.join(OUT_DIR, "train.json"), train_docs)
    save_json(os.path.join(OUT_DIR, "dev.json"), dev_docs)
    save_json(os.path.join(OUT_DIR, "test.json"), test_docs)

    print(f"Train: {len(train_docs)}")
    print(f"Dev:   {len(dev_docs)}")
    print(f"Test:  {len(test_docs)}")


if __name__ == "__main__":
    main()