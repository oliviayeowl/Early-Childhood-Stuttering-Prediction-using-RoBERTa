import os
import json
import torch
import pandas as pd
from transformers import (
    RobertaTokenizerFast, 
    RobertaForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset

class StutterModelSystem:
    """
    Main system for RoBERTa model training, hybrid post-processing, and evaluation.
    """
    def __init__(self, model_name="roberta-base"):
        self.model_name = model_name
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
        self.label_list = []
        self.label2id = {}
        self.id2label = {}

    def prepare_labels(self, train_data_path):
        """Extracts unique labels from the training data to map them to IDs."""
        unique_labels = set()
        with open(train_data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                unique_labels.update(data['labels'])
        
        self.label_list = sorted(list(unique_labels))
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for i, l in enumerate(self.label_list)}
        return len(self.label_list)

    def tokenize_and_align_labels(self, examples):
        """Tokenizes text and aligns BIO labels with sub-word tokens."""
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100) # Ignore special tokens
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id[label[word_idx]])
                else:
                    # For sub-words, we can either use the same label or -100
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def train_roberta(self, train_data_path, output_dir="./stutter_model"):
        """Performs actual RoBERTa training."""
        num_labels = self.prepare_labels(train_data_path)
        
        # Load dataset
        data = []
        with open(train_data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        raw_dataset = Dataset.from_list(data)
        tokenized_dataset = raw_dataset.map(self.tokenize_and_align_labels, batched=True)

        # Initialize Model
        model = RobertaForTokenClassification.from_pretrained(
            self.model_name, num_labels=num_labels, label2id=self.label2id, id2label=self.id2label
        )

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="no", # Can change to "epoch" if you have a val set
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
        )

        print("Starting Training...")
        trainer.train()
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")

    # --- HYBRID SCRIPT: THE SHRINKER ---
    def bio_to_salt(self, tokens_with_labels):
        """Converts BIO labels back into SALT tags with automatic counting."""
        result = []
        i = 0
        while i < len(tokens_with_labels):
            word, label = tokens_with_labels[i]
            if label.startswith('B-'):
                tag_type = label.split('-')[1]
                count = 0
                j = i + 1
                while j < len(tokens_with_labels) and tokens_with_labels[j][1] == f'I-{tag_type}':
                    count += 1
                    j += 1
                if tag_type in ['WW', 'I', 'P']:
                    result.append(f"[^ {tag_type}{count}]")
                    result.append(word)
                    i = j 
                else:
                    result.append(f"[^ {tag_type.lower()}]")
                    result.append(word)
                    i += 1
            else:
                result.append(word)
                i += 1
        return " ".join(result)

    def evaluate_on_test_set(self, test_csv_path, model_path="./stutter_model"):
        """Tests the model on your raw test set and calculates accuracy."""
        # Load the trained model
        model = RobertaForTokenClassification.from_pretrained(model_path)
        df_test = pd.read_csv(test_csv_path)
        
        all_results = []
        
        print(f"Evaluating on {len(df_test)} test lines...")
        for _, row in df_test.iterrows():
            raw_text = str(row['raw_transcript'])
            
            # 1. Inference
            inputs = self.tokenizer(raw_text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            predictions = torch.argmax(logits, dim=2)
            
            # 2. Map predictions back to words
            word_ids = inputs.word_ids()
            tokens = raw_text.split()
            predicted_labels = []
            
            # Simple alignment for demo/undergrad scope
            current_word_idx = -1
            for idx, word_id in enumerate(word_ids):
                if word_id is not None and word_id != current_word_idx:
                    label_id = predictions[0][idx].item()
                    predicted_labels.append(self.id2label[label_id])
                    current_word_idx = word_id
            
            # 3. Hybrid Post-processing
            prediction_pairs = list(zip(tokens, predicted_labels))
            final_salt = self.bio_to_salt(prediction_pairs)
            all_results.append(final_salt)
            
        df_test['model_prediction'] = all_results
        df_test.to_csv("final_research_results.csv", index=False)
        print("Evaluation complete. Results saved to final_research_results.csv")

if __name__ == "__main__":
    # CONFIGURATION
    TRAIN_DATA = "/Users/wanlingyeo/URECA/training_data.jsonl"
    TEST_DATA = "/Users/wanlingyeo/URECA/test_set_evaluation.csv"
    
    system = StutterModelSystem()
    
    # To run this, you need: pip install transformers datasets torch pandas
    if os.path.exists(TRAIN_DATA):
        system.train_roberta(TRAIN_DATA)
        if os.path.exists(TEST_DATA):
            system.evaluate_on_test_set(TEST_DATA)
    else:
        print("Error: training_data.jsonl not found. Run preprocessing.py first!")
