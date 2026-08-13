import os
import re
import random
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data_repo/data"
OUTPUT_DIR = "data_repo/speaker_split"

SEED = 42

FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet"
}


# ============================================================
# GET GROUP / SPEAKER ID
# ============================================================

def get_group_id(path):

    path = str(path).replace("\\", "/")

    # --------------------------------------------------------
    # 1. speaker_###
    # --------------------------------------------------------

    match = re.search(
        r"(speaker_\d+)",
        path,
        re.IGNORECASE
    )

    if match:
        return "speaker:" + match.group(1).lower()


    # --------------------------------------------------------
    # 2. Mozilla_dataset/<hash>
    # --------------------------------------------------------

    if "Mozilla_dataset/" in path:

        parts = path.split("/")

        try:
            idx = parts.index("Mozilla_dataset")

            if idx + 1 < len(parts):

                group = parts[idx + 1]

                return "mozilla:" + group.lower()

        except ValueError:
            pass


    # --------------------------------------------------------
    # 3. Mozilla_cloned/<hash>
    # --------------------------------------------------------

    if "Mozilla_cloned/" in path:

        parts = path.split("/")

        try:
            idx = parts.index("Mozilla_cloned")

            if idx + 1 < len(parts):

                group = parts[idx + 1]

                return "mozilla:" + group.lower()

        except ValueError:
            pass


    # --------------------------------------------------------
    # 4. self_documented/<name>
    # --------------------------------------------------------

    if "self_documented/" in path:

        parts = path.split("/")

        try:
            idx = parts.index("self_documented")

            if idx + 1 < len(parts):

                person = parts[idx + 1]

                return "self:" + person.lower()

        except ValueError:
            pass


    # --------------------------------------------------------
    # 5. Punjabi
    # --------------------------------------------------------

    if "punjabi" in path.lower():

        filename = os.path.basename(path)

        filename = os.path.splitext(filename)[0]

        # Remove clone / cloned prefix
        filename = re.sub(
            r"^(clone|cloned)[_-]?",
            "",
            filename,
            flags=re.IGNORECASE
        )

        # Remove sample number
        filename = re.sub(
            r"[_-]\d+$",
            "",
            filename
        )

        filename = filename.strip()

        if filename:

            return "punjabi:" + filename.lower()


    # --------------------------------------------------------
    # 6. Unknown
    #
    # DO NOT group unknown files together.
    # Give each file its own group.
    # --------------------------------------------------------

    return "unknown:" + path


# ============================================================
# LOAD ALL PARQUETS
# ============================================================

print("Loading parquet files...")

dfs = []

for split, filename in FILES.items():

    path = os.path.join(
        DATA_DIR,
        filename
    )

    df = pd.read_parquet(path)

    print(
        f"{filename}: {len(df)} samples"
    )

    dfs.append(df)


df = pd.concat(
    dfs,
    ignore_index=True
)

print(
    "\nTotal samples:",
    len(df)
)


# ============================================================
# CREATE GROUP IDs
# ============================================================

print("\nFinding speakers/groups...")

df["group_id"] = df["path"].apply(
    get_group_id
)


print(
    "Total groups:",
    df["group_id"].nunique()
)


# ============================================================
# GROUP TYPE COUNTS
# ============================================================

print("\nGroup types:")

group_types = (
    df["group_id"]
    .str.split(":")
    .str[0]
    .value_counts()
)

print(group_types)


# ============================================================
# CHECK UNKNOWN
# ============================================================

unknown = df[
    df["group_id"].str.startswith("unknown:")
]

print(
    "\nUnknown samples:",
    len(unknown)
)


# ============================================================
# SHUFFLE GROUPS
# ============================================================

groups = list(
    df["group_id"].unique()
)

random.seed(SEED)

random.shuffle(groups)


# ============================================================
# 70 / 15 / 15 GROUP SPLIT
# ============================================================

n_groups = len(groups)

n_train = int(
    n_groups * 0.70
)

n_val = int(
    n_groups * 0.15
)


train_groups = set(
    groups[:n_train]
)

val_groups = set(
    groups[
        n_train:n_train + n_val
    ]
)

test_groups = set(
    groups[
        n_train + n_val:
    ]
)


print("\nSpeaker/group split:")

print(
    "Train groups:",
    len(train_groups)
)

print(
    "Validation groups:",
    len(val_groups)
)

print(
    "Test groups:",
    len(test_groups)
)


# ============================================================
# CREATE SPLITS
# ============================================================

train_df = df[
    df["group_id"].isin(train_groups)
].copy()

val_df = df[
    df["group_id"].isin(val_groups)
].copy()

test_df = df[
    df["group_id"].isin(test_groups)
].copy()


# ============================================================
# LEAKAGE CHECK BEFORE SAVING
# ============================================================

train_group_check = set(
    train_df["group_id"]
)

val_group_check = set(
    val_df["group_id"]
)

test_group_check = set(
    test_df["group_id"]
)


train_val = (
    train_group_check &
    val_group_check
)

train_test = (
    train_group_check &
    test_group_check
)

val_test = (
    val_group_check &
    test_group_check
)


print("\n" + "=" * 60)
print("GROUP LEAKAGE CHECK")
print("=" * 60)

print(
    "Train ∩ Validation:",
    len(train_val)
)

print(
    "Train ∩ Test:",
    len(train_test)
)

print(
    "Validation ∩ Test:",
    len(val_test)
)


# ============================================================
# STOP IF LEAKAGE EXISTS
# ============================================================

if (
    len(train_val) > 0
    or
    len(train_test) > 0
    or
    len(val_test) > 0
):

    print(
        "\nERROR: GROUP LEAKAGE FOUND!"
    )

    print(
        "Files were NOT saved."
    )

    raise SystemExit


print(
    "\nNo group leakage detected."
)


# ============================================================
# DROP HELPER COLUMN
# ============================================================

train_df = train_df.drop(
    columns=["group_id"]
)

val_df = val_df.drop(
    columns=["group_id"]
)

test_df = test_df.drop(
    columns=["group_id"]
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


train_df.to_parquet(
    os.path.join(
        OUTPUT_DIR,
        "train.parquet"
    ),
    index=False
)

val_df.to_parquet(
    os.path.join(
        OUTPUT_DIR,
        "validation.parquet"
    ),
    index=False
)

test_df.to_parquet(
    os.path.join(
        OUTPUT_DIR,
        "test.parquet"
    ),
    index=False
)


# ============================================================
# PRINT SAMPLE COUNTS
# ============================================================

print("\n" + "=" * 60)
print("SAVED DATASET")
print("=" * 60)

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nTRAIN labels:")
print(
    train_df["label"].value_counts()
)

print("\nVALIDATION labels:")
print(
    val_df["label"].value_counts()
)

print("\nTEST labels:")
print(
    test_df["label"].value_counts()
)


# ============================================================
# FINAL PATH
# ============================================================

print(
    "\nSaved files:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "train.parquet"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "validation.parquet"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "test.parquet"
    )
)