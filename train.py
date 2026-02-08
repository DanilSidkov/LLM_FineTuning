import torch
import argparse
import yaml
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset, DatasetDict
from peft import get_peft_model, PromptEncoderConfig, TaskType, PeftModel
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class SentimentPTSystem:
    def __init__(self, config_path="config/training_config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_enviroment()

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def setup_enviroment(self):
        torch.manual_seed(self.config["training"]["seed"])
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and not self.config["training"]["use_cpu"]
            else "cpu"
        )

        if self.config["logging"]["use_wandb"]:
            wandb.init(
                project=self.config["logging"]["wandb_project"], config=self.config
            )

    def load_and_preprocess_data(self):
        data_config = self.config["data"]

        if data_config["source"] == "csv":
            df = pd.read_csv(data_config["path"])

            required_cols = ["text", "label"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Column {col} not found in dataset")

            if self.config["model"]["num_labels"] == 2:
                label_map = data_config.get("label_map", {"negative": 0, "positive": 1})
                df["label"] = df["label"].map(label_map)
            else:
                unique_labels = df["label"].unique()
                label_to_id = {
                    label: i for i, label in enumerate(sorted(unique_labels))
                }
                df["label"] = df["label"].map(label_to_id)

            train_df, temp_df = train_test_split(
                df,
                test_size=data_config["test_size"] + data_config["val_size"],
                random_state=self.config["training"]["seed"],
            )

            val_size = data_config["val_size"] / (
                data_config["test_size"] + data_config["val_size"]
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_size,
                random_state=self.config["training"]["seed"],
            )

            train_dataset = Dataset.from_pandas(train_df)
            val_dataset = Dataset.from_pandas(val_df)
            test_dataset = Dataset.from_pandas(test_df)

            self.dataset = DatasetDict(
                {
                    "train": train_dataset,
                    "validation": val_dataset,
                    "test": test_dataset,
                }
            )

        elif data_config["source"] == "huggingface":
            from datasets import load_dataset

            self.dataset = load_dataset(
                data_config["path"], split=data_config.get("split", "train")
            )

            if data_config.get("preprocess", False):
                self.dataset = self.dataset.map(self.preprocess_funtion, batched=True)

        print(f"Dataset loaded: {self.dataset}")
        print(f"Label distribution: {self.dataset['train'].unique('label')}")

    def preprocess_funtion(self, examples):
        tokenizer = self.tokenizer

        if self.config["model"].get("prompt_template"):
            template = self.config["model"]["prompt_template"]
            texts = [template.format(text=text) for text in examples["text"]]
        else:
            texts = examples["text"]

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.config["model"]["max_length"],
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    def initialize_model_and_tokenizer(self):
        model_config = self.config["model"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["bas_model"], use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_config["base_model"],
            num_labels=model_config["num_labels"],
            ignore_mismatched_sizes=True,
        )

        peft_config = PromptEncoderConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=model_config["num_virtual_tokens"],
            encoder_hidden_size=model_config["encoder_hidden_size"],
            encoder_num_layers=model_config["encoder_num_layers"],
            encoder_dropout=model_config["encoder_dropout"],
            prompt_tuning_init=model_config["prompt_tuning_init"],
            prompt_tuning_init_text=model_config.get(
                "prompt_tuning_init_text", "Classify the sentiment of this text:"
            ),
            tokenizer_name_or_path=model_config["base_model"],
        )

        self.model = get_peft_model(self.base_model, peft_config)
        self.model.print_trainable_parameters()
        self.model = self.model.to(self.device)

        print(f"Model initialized with P-Tuning v2")
        print(
            f"Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )
        print(f"Total params: {sum(p.numel() for p in self.model.parameters())}")

    def get_training_arguments(self):
        training_config = self.config["training"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{self.config['experiment']['name']}_{timestamp}"

        output_dir = Path(self.config["experiment"]["output_dir"]) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            num_train_epochs=training_config["num_epochs"],
            per_device_train_batch_size=training_config["batch_size"],
            per_device_eval_batch_size=training_config["batch_size"],
            gradient_accumulation_steps=training_config.get(
                "gradient_accumulation_steps", 1
            ),
            warmup_ratio=training_config["warmup_ratio"],
            weight_decay=training_config["weight_decay"],
            learning_rate=training_config["learning_rate"],
            logging_dir=str(output_dir / "logs"),
            logging_steps=training_config["logging_steps"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.config["logging"]["use_wandb"] else "none",
            fp16=torch.cuda.is_available() and training_config.get("fp16", False),
            push_to_hub=False,
            seed=self.config["training"]["seed"],
            data_seed=self.config["training"]["seed"],
        )

        return training_args

    def compute_metrics(self, eval_pred):
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def train(self):
        self.load_and_preprocess_data()

        tokenized_datasets = self.dataset.map(
            self.preprocess_funtion,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )

        self.initialize_model_and_tokenizer()

        training_args = self.get_training_arguments()

        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.config["model"]["max_length"],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        train_result = trainer.train()

        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)

        test_metrics = trainer.evaluate(tokenized_datasets["test"])

        metrics_path = Path(training_args.output_dir) / "metrics.json"
        import json

        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)

        self.visualize_prompts(training_args.output_dir)

        return train_result, test_metrics

    def visualize_prompts(self, output_dir):
        import matplotlib.pyplot as plt
        import numpy as np

        prompt_encoder = self.model.prompt_encoder
        if hasattr(prompt_encoder, "embedding"):
            prompt_embeddings = prompt_encoder.embedding.weight.detach().cpu().numpy()

            plt.figure(figsize=(12, 6))
            plt.imshow(prompt_embeddings, aspect="auto", cmap="RdYlBu")
            plt.colorbar(label="Embedding Value")
            plt.title("P-Tuning v2 Prompt Embeddings")
            plt.xlabel("Hidden Dimension")
            plt.ylabel("Virtual Token Index")

            viz_path = Path(output_dir) / "prompt_embeddings.png"
            plt.savefig(viz_path, dpi=150, bbox_inches="tight")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="P-Tuning v2 для Sentiment Analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--upload",
        type=str,
        default=None,
        help="HuggingFace repository name for upload",
    )

    args = parser.parse_args()
    
    system = SentimentPTSystem(config_path=args.config)

    train_result, test_metrics = system.train()

if __name__ == "__main__":
    main()
