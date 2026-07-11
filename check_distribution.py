import pandas as pd

df = pd.read_csv("dreaddit-train.csv")

stress_keywords = [
    "anxious",
    "anxiety",
    "panic",
    "depressed",
    "overwhelmed",
    "stress",
    "stressed",
    "pressure",
    "worried",
    "fear",
    "afraid",
    "nervous",
    "exhausted",
    "burnout",
    "tired"
]

def assign_stress_level(text, label):

    if label == 0:
        return 0

    text = str(text).lower()

    score = sum(keyword in text for keyword in stress_keywords)

    if score <= 1:
        return 1
    else:
        return 2

df["stress_level"] = df.apply(
    lambda row: assign_stress_level(row["text"], row["label"]),
    axis=1
)

print(df["stress_level"].value_counts())