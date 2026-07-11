import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# PHASE 1: LOAD DATASET
# ==========================================

df = pd.read_csv("dreaddit-train.csv")

# ==========================================
# PHASE 2: CREATE 3 STRESS LEVELS
# ==========================================

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

    # Original No-Stress
    if label == 0:
        return 0  # Low Stress

    text = str(text).lower()

    score = sum(
        keyword in text
        for keyword in stress_keywords
    )

    if score <= 1:
        return 1  # Mild Stress
    else:
        return 2  # High Stress

df["stress_level"] = df.apply(
    lambda row: assign_stress_level(
        row["text"],
        row["label"]
    ),
    axis=1
)
# ==========================================
# PHASE 3: PREPARE DATA
# ==========================================

texts = df["text"].astype(str).values
labels = df["stress_level"].values

print("\nClass Distribution:")
print(df["stress_level"].value_counts())

# ==========================================
# PHASE 4: TOKENIZATION
# ==========================================

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

encodings = tokenizer.batch_encode_plus(
    texts.tolist(),
    add_special_tokens=True,
    max_length=128,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt"
)

input_ids = encodings["input_ids"]
attention_masks = encodings["attention_mask"]

labels_tensor = torch.tensor(labels)

# ==========================================
# PHASE 5: TRAIN / VALIDATION SPLIT
# ==========================================

train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    input_ids,
    labels_tensor,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_masks, val_masks, _, _ = train_test_split(
    attention_masks,
    labels_tensor,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ==========================================
# PHASE 6: DATA LOADERS
# ==========================================

train_dataset = TensorDataset(
    train_inputs,
    train_masks,
    train_labels
)

val_dataset = TensorDataset(
    val_inputs,
    val_masks,
    val_labels
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16
)

# ==========================================
# PHASE 7: LOAD BERT
# ==========================================

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

# ==========================================
# PHASE 8: TRAINING
# ==========================================

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

epochs = 3

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch in train_loader:

        b_input_ids, b_masks, b_labels = [
            x.to(device) for x in batch
        ]

        optimizer.zero_grad()

        outputs = model(
            b_input_ids,
            attention_mask=b_masks,
            labels=b_labels
        )

        loss = outputs.loss

        total_loss += loss.item()

        loss.backward()

        optimizer.step()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch+1}/{epochs} | Loss = {avg_loss:.4f}"
    )

# ==========================================
# PHASE 9: EVALUATION
# ==========================================

model.eval()

predictions = []
actuals = []

with torch.no_grad():

    for batch in val_loader:

        b_input_ids, b_masks, b_labels = [
            x.to(device) for x in batch
        ]

        outputs = model(
            b_input_ids,
            attention_mask=b_masks
        )

        preds = torch.argmax(
            outputs.logits,
            dim=1
        )

        predictions.extend(
            preds.cpu().numpy()
        )

        actuals.extend(
            b_labels.cpu().numpy()
        )

accuracy = accuracy_score(
    actuals,
    predictions
)

print("\nValidation Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(
    classification_report(
        actuals,
        predictions,
        target_names=[
            "Low Stress",
            "Mild Stress",
            "High Stress"
        ]
    )
)

# ==========================================
# PHASE 10: SAVE MODEL
# ==========================================

SAVE_PATH = "./bert_stress_model_v2"

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(
    f"\n3-Class BERT Model saved to: {SAVE_PATH}"
)