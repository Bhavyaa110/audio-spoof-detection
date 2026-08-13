import pandas as pd
import re

files = {
    "TRAIN": "train-00000-of-00001.parquet",
    "VALIDATION": "validation-00000-of-00001.parquet",
    "TEST": "test-00000-of-00001.parquet"
}

speakers = {}

for split, file in files.items():

    df = pd.read_parquet(
        "data_repo/data/" + file
    )

    paths = df["path"].astype(str)

    found = set()

    for path in paths:

        matches = re.findall(
            r"speaker_\d+",
            path,
            flags=re.IGNORECASE
        )

        for speaker in matches:
            found.add(
                speaker.lower()
            )

    speakers[split] = found

    print(
        f"{split} speakers: {len(found)}"
    )


train = speakers["TRAIN"]
val = speakers["VALIDATION"]
test = speakers["TEST"]


train_val = train & val
train_test = train & test
val_test = val & test


print("\n" + "=" * 50)
print("SPEAKER OVERLAP CHECK")
print("=" * 50)

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


print("\nExamples:")

print(
    "Train ∩ Validation:",
    list(train_val)[:20]
)

print(
    "Train ∩ Test:",
    list(train_test)[:20]
)

print(
    "Validation ∩ Test:",
    list(val_test)[:20]
)