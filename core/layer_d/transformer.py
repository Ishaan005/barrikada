import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.layer_c.training_utils import load_data
from core.settings import Settings

from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


settings = Settings()



def main():
    X, y, df = load_data(settings.dataset_path)

    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_ds = Dataset.from_dict(
        {
            "text": df_train["processed_text"].tolist(),
            "labels": df_train["label"].astype(int).tolist(),
        }
    )
    test_ds = Dataset.from_dict(
        {
            "text": df_test["processed_text"].tolist(),
            "labels": df_test["label"].astype(int).tolist(),
        }
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])
    train_ds.set_format("torch")
    test_ds.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./outputs",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
    )

    trainer.train()

    results = trainer.evaluate()
    print("Evaluation results:", results)

if __name__ == "__main__":
    main()
