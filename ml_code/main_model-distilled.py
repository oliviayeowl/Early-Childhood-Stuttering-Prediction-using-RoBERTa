import os
import json
import torch
import pandas as pd
import shutil
import re
from transformers import (
    RobertaTokenizerFast, 
    RobertaForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support

class StutterModelSystem:
    """
    Main system for RoBERTa model training, hybrid post-processing, and evaluation.
    Includes full metrics suite for Precision, Recall, and F1-score.
    """
    def __init__(self, model_name="distilroberta-base"): 
        self.model_name = model_name
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
        self.label_list = []
        self.label2id = {}
        self.id2label = {}

    def prepare_labels(self, train_data_path):
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
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def train_roberta(self, train_data_path, output_dir="./stutter_model"):
        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        if os.path.exists('./logs'): shutil.rmtree('./logs')

        num_labels = self.prepare_labels(train_data_path)
        data = []
        with open(train_data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        raw_dataset = Dataset.from_list(data)
        tokenized_dataset = raw_dataset.map(self.tokenize_and_align_labels, batched=True)

        model = RobertaForTokenClassification.from_pretrained(
            self.model_name, num_labels=num_labels, label2id=self.label2id, id2label=self.id2label
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            weight_decay=0.01,
            save_strategy="no", 
            save_total_limit=1,
            logging_steps=100,
            report_to="none" 
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
        )

        print(f"Starting Training with {self.model_name}...")
        trainer.train()
        trainer.save_model(output_dir)
        print(f"Final model saved to {output_dir}")

    def bio_to_salt(self, tokens_with_labels):
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

    def extract_tags(self, salt_string):
        """Extracts tags and their following words for matching."""
        # Finds [^ TAG#] Word or [^ tag] Word
        matches = re.findall(r'\[\^\s*([A-Za-z<>+/]+)(\d*)\s*\]\s*(\S+)', salt_string)
        # Returns list of (tag_type, count, word)
        return [(m[0].upper(), m[1], m[2].lower()) for m in matches]

    def calculate_metrics(self, df):
        """Calculates Precision, Recall, and F1 based on tag matches."""
        total_actual = 0
        total_predicted = 0
        correct_matches = 0 # Strict match: Type + Word + Count
        partial_matches = 0 # Partial match: Type + Word (Count can differ)

        for _, row in df.iterrows():
            actual_tags = self.extract_tags(str(row['ground_truth_salt']))
            predicted_tags = self.extract_tags(str(row['model_prediction']))
            
            total_actual += len(actual_tags)
            total_predicted += len(predicted_tags)
            
            # Simple matching logic
            for p_tag, p_count, p_word in predicted_tags:
                for a_tag, a_count, a_word in actual_tags:
                    if p_tag == a_tag and p_word == a_word:
                        partial_matches += 1
                        if p_count == a_count:
                            correct_matches += 1
                        break
        
        precision = correct_matches / total_predicted if total_predicted > 0 else 0
        recall = correct_matches / total_actual if total_actual > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Partial metrics
        p_precision = partial_matches / total_predicted if total_predicted > 0 else 0
        p_recall = partial_matches / total_actual if total_actual > 0 else 0
        p_f1 = 2 * (p_precision * p_recall) / (p_precision + p_recall) if (p_precision + p_recall) > 0 else 0

        return {
            "Strict": {"Precision": precision, "Recall": recall, "F1": f1},
            "Partial": {"Precision": p_precision, "Recall": p_recall, "F1": p_f1}
        }

    def evaluate_on_test_set(self, test_csv_path, model_path="./stutter_model"):
        if not os.path.exists(model_path):
            print("Model not found.")
            return

        model = RobertaForTokenClassification.from_pretrained(model_path)
        df_test = pd.read_csv(test_csv_path)
        all_results = []
        
        print(f"Evaluating on {len(df_test)} test lines...")
        model.eval()
        for _, row in df_test.iterrows():
            raw_text = str(row['raw_transcript'])
            inputs = self.tokenizer(raw_text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            predictions = torch.argmax(logits, dim=2)
            
            word_ids = inputs.word_ids()
            tokens = raw_text.split()
            predicted_labels = []
            current_word_idx = -1
            for idx, word_id in enumerate(word_ids):
                if word_id is not None and word_id != current_word_idx:
                    label_id = predictions[0][idx].item()
                    predicted_labels.append(self.id2label[label_id])
                    current_word_idx = word_id
            
            prediction_pairs = list(zip(tokens, predicted_labels))
            final_salt = self.bio_to_salt(prediction_pairs)
            all_results.append(final_salt)
            
        df_test['model_prediction'] = all_results
        
        # Calculate Research Metrics
        metrics = self.calculate_metrics(df_test)
        
        # Save results and print report
        df_test.to_csv("final_research_results.csv", index=False)
        
        report = f"""
=========================================
RESEARCH EVALUATION REPORT
=========================================
Model: {self.model_name}
Test Set: {test_csv_path} ({len(df_test)} utterances)

STRICT METRICS (Type + Word + Count must match):
- Precision: {metrics['Strict']['Precision']:.4f}
- Recall:    {metrics['Strict']['Recall']:.4f}
- F1-Score:  {metrics['Strict']['F1']:.4f}

PARTIAL METRICS (Type + Word match, Count ignored):
- Precision: {metrics['Partial']['Precision']:.4f}
- Recall:    {metrics['Partial']['Recall']:.4f}
- F1-Score:  {metrics['Partial']['F1']:.4f}
=========================================
"""
        print(report)
        with open("evaluation_report.txt", "w") as f:
            f.write(report)
        print("Success! Results saved to final_research_results.csv and evaluation_report.txt")

if __name__ == "__main__":
    TRAIN_DATA = "training_data.jsonl"
    TEST_DATA = "test_set_evaluation.csv"
    system = StutterModelSystem()
    if os.path.exists(TRAIN_DATA):
        try:
            system.train_roberta(TRAIN_DATA)
            if os.path.exists(TEST_DATA):
                system.evaluate_on_test_set(TEST_DATA)
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Error: training_data.jsonl not found.")
