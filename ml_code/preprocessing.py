import re
import json
import os
import pandas as pd
import random

class SALTPreprocessor:
    """
    Handles the expansion of SALT/CHAT data into BIO format for RoBERTa training,
    generates a raw-ified test set from the 15th dataset, and creates a full 
    tagging_check.csv for manual verification.
    """
    def __init__(self):
        # Improved pattern to capture tags, phrases, and standalone markers
        self.token_pattern = r'\[\^[^\]]+\]|<[^>]+>|\[/\]|\+\.\.\.|\+/|\S+'
        self.punctuation_to_strip = r'[.?!\[\]^]' 
        
        self.fine_label_map = {
            'XXX': 'unintelligible',
            '+': 'shortened/incomplete sentence',
            'I': 'interjection',
            'PW': 'part-word repetition',
            'WW': 'single-syllable word repetition',
            'DP': 'dysrhythmic phonation',
            'R': 'restart/revision',
            '<>': 'abandoned utterance',
            'P': 'phrase repetition',
            '/': 'restarting a sentence'
        }
        self.sld_list = ['PW', 'WW', 'DP', 'P']

    def get_td_sld(self, tag):
        return 'SLD' if tag in self.sld_list else 'TD'

    def salt_to_bio(self, line):
        """Converts SALT line to BIO format (Used for Training Data and Full Check)."""
        if not line.startswith('*CHI:'): return []
        
        line = re.sub(r'^\*CHI:\s+', '', line)
        line = re.sub(r'%.*$', '', line).strip()
        
        # Split +/ and +... specifically
        line = re.sub(r'\+/', r' + / ', line)
        line = re.sub(r'\+\.\.\.', r' + xxx ', line) 
        
        # Ensure standalone + and / have spaces
        line = re.sub(r'(?<!\[)\+(?!\]|\.\.\.|/)', r' + ', line)
        line = re.sub(r'(?<!\[|/)/(?!\])', r' / ', line)
        
        raw_tokens = re.findall(self.token_pattern, line)
        results = []
        i = 0
        while i < len(raw_tokens):
            token = raw_tokens[i]
            
            if token.startswith('[^'):
                match = re.search(r'\[\^\s*([a-z<>+/]+)\s*(\d*)\]', token.lower())
                if match:
                    tag_type = match.group(1).upper()
                    count = int(match.group(2)) if match.group(2) else 0
                    i += 1
                    if i < len(raw_tokens):
                        target = raw_tokens[i]
                        clean_target = re.sub(r'[<>]', '', target)
                        target_words = [re.sub(self.punctuation_to_strip, '', w) for w in clean_target.split() if w]
                        
                        # Handle specific <> tag
                        current_tag = tag_type if tag_type != '<>' else '<>'
                        
                        for idx, w in enumerate(target_words):
                            label_prefix = "B-" if idx == 0 else "I-"
                            results.append({
                                "word": w, 
                                "BIO_label": f"{label_prefix}{current_tag}", 
                                "fine_label": self.fine_label_map.get(current_tag, current_tag), 
                                "TDorSLD": self.get_td_sld(current_tag)
                            })
                            if current_tag in ['WW', 'I', 'P'] and count > 0:
                                for _ in range(count):
                                    results.append({
                                        "word": w, 
                                        "BIO_label": f"I-{current_tag}", 
                                        "fine_label": self.fine_label_map.get(current_tag, current_tag),
                                        "TDorSLD": self.get_td_sld(current_tag)
                                    })
                i += 1
            elif token.startswith('<'):
                phrase = token[1:-1]
                i += 1
                if i < len(raw_tokens) and raw_tokens[i] == '[/]':
                    # It's a revision [/]
                    words = [re.sub(self.punctuation_to_strip, '', x) for x in phrase.split() if x]
                    for idx, w in enumerate(words):
                        label_prefix = "B-" if idx == 0 else "I-"
                        results.append({"word": w, "BIO_label": f"{label_prefix}R", "fine_label": self.fine_label_map.get('R'), "TDorSLD": "TD"})
                    i += 1
                else:
                    # It's an abandoned utterance <>
                    words = [re.sub(self.punctuation_to_strip, '', x) for x in phrase.split() if x]
                    for idx, w in enumerate(words):
                        label_prefix = "B-" if idx == 0 else "I-"
                        results.append({"word": w, "BIO_label": f"{label_prefix}<>", "fine_label": self.fine_label_map.get('<>'), "TDorSLD": "TD"})
            else:
                clean_word = re.sub(r'[&~]', '', token)
                clean_word = re.sub(self.punctuation_to_strip, '', clean_word)
                if clean_word:
                    if clean_word.lower() == 'xxx':
                        results.append({"word": clean_word, "BIO_label": "B-XXX", "fine_label": self.fine_label_map.get('XXX'), "TDorSLD": "TD"})
                    elif clean_word in ['+', '/']:
                        results.append({"word": clean_word, "BIO_label": f"B-{clean_word}", "fine_label": self.fine_label_map.get(clean_word), "TDorSLD": "TD"})
                    else:
                        results.append({"word": clean_word, "BIO_label": "O", "fine_label": "fluent", "TDorSLD": "N/A"})
                i += 1
        return results

    def salt_to_raw(self, line):
        """Converts SALT line to RAW text (Used for Test Set)."""
        if not line.startswith('*CHI:'): return None
        line = re.sub(r'^\*CHI:\s+', '', line)
        line = re.sub(r'%.*$', '', line).strip()
        
        raw_tokens = re.findall(self.token_pattern, line)
        raw_words = []
        i = 0
        while i < len(raw_tokens):
            token = raw_tokens[i]
            if token.startswith('[^'):
                match = re.search(r'\[\^\s*([a-z<>+/]+)\s*(\d*)\]', token.lower())
                if match:
                    tag_type = match.group(1).upper()
                    count = int(match.group(2)) if match.group(2) else 0
                    i += 1
                    if i < len(raw_tokens):
                        target = re.sub(r'[<>]', '', raw_tokens[i])
                        if tag_type == 'I':
                            raw_words.extend(["uh"] * count)
                        elif tag_type in ['WW', 'P']:
                            target_words = target.split()
                            for _ in range(count + 1):
                                raw_words.extend(target_words)
                        elif tag_type == 'PW':
                            word = target.split()[0]
                            prefix = word[0] + "-" if word else ""
                            raw_words.append(prefix + word)
                        else:
                            raw_words.extend(target.split())
                i += 1
            elif token.startswith('<'):
                phrase = token[1:-1]
                i += 1
                if i < len(raw_tokens) and raw_tokens[i] == '[/]':
                    raw_words.extend(phrase.split())
                    raw_words.extend(phrase.split())
                    i += 1
                else:
                    raw_words.extend(phrase.split())
            else:
                clean = re.sub(r'[&~]', '', token)
                if clean: raw_words.append(clean)
                i += 1
        return " ".join([re.sub(self.punctuation_to_strip, '', w) for w in raw_words if w])

    def process_all(self, input_dir):
        filenames = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt") or f.endswith(".cha")])
        if not filenames: return
        
        train_files = filenames[:-1] if len(filenames) > 1 else filenames
        test_file = filenames[-1] if len(filenames) > 1 else None
        
        train_data = []
        all_pairs_for_check = []
        
        for fname in filenames:
            with open(os.path.join(input_dir, fname), 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    bio = self.salt_to_bio(line)
                    if bio:
                        if fname in train_files:
                            train_data.append({"tokens": [p["word"] for p in bio], "labels": [p["BIO_label"] for p in bio]})
                        for entry in bio:
                            all_pairs_for_check.append({
                                "filename": fname,
                                "line_number": line_num,
                                "word": entry["word"],
                                "BIO_label": entry["BIO_label"],
                                "fine_label": entry["fine_label"],
                                "TDorSLD": entry["TDorSLD"]
                            })
        
        with open("training_data.jsonl", 'w') as f:
            for entry in train_data: f.write(json.dumps(entry) + '\n')
            
        df_all = pd.DataFrame(all_pairs_for_check)
        df_all.to_csv("tagging_check.csv", index=False)
        self.generate_stats(df_all)

        if test_file:
            test_lines = []
            with open(os.path.join(input_dir, test_file), 'r', encoding='utf-8', errors='ignore') as f:
                all_chi_lines = [l for l in f if l.startswith('*CHI:')]
                selected = random.sample(all_chi_lines, min(300, len(all_chi_lines)))
                for line in selected:
                    raw_text = self.salt_to_raw(line)
                    bio_truth = self.salt_to_bio(line)
                    if raw_text and bio_truth:
                        test_lines.append({
                            "raw_transcript": raw_text,
                            "ground_truth_salt": line.replace('*CHI:', '').strip(),
                            "ground_truth_bio": bio_truth
                        })
            pd.DataFrame(test_lines).to_csv("test_set_evaluation.csv", index=False)
        print(f"Full checkable CSV generated: tagging_check.csv")

    def generate_stats(self, df):
        """Generates statistical breakdown of disfluencies and categories."""
        total_tokens = len(df)
        
        # Identify types by stripping B- and I-
        df['tag_type'] = df['BIO_label'].apply(lambda x: x.split('-')[1] if '-' in x else 'O')
        
        stats = []
        target_tags = ['XXX', '+', 'I', 'PW', 'WW', 'DP', 'R', '<>', 'P', '/']
        
        print("\n" + "="*40)
        print("DATASET STATISTICS REPORT")
        print("="*40)
        
        for tag in target_tags:
            # COUNTING LOGIC: We only count the "B-" labels to count the event once
            count = len(df[(df['tag_type'] == tag) & (df['BIO_label'].str.startswith('B-'))])
            percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
            stats.append({
                "Category": f"Disfluency: {tag}",
                "Description": self.fine_label_map.get(tag, "N/A"),
                "Count": count,
                "Percentage": f"{percentage:.2f}%"
            })
            print(f"{tag:5} ({self.fine_label_map.get(tag, 'N/A'):<30}): {count:<6} ({percentage:.2f}%)")

        print("-" * 40)
        
        fluent_count = len(df[df['BIO_label'] == 'O'])
        fluent_pct = (fluent_count / total_tokens) * 100 if total_tokens > 0 else 0
        stats.append({"Category": "High-Level: Fluent", "Description": "Fluent speech", "Count": fluent_count, "Percentage": f"{fluent_pct:.2f}%"})
        print(f"{'Fluent':<37}: {fluent_count:<6} ({fluent_pct:.2f}%)")

        # TD and SLD counts also only count B- labels to represent event counts
        td_count = len(df[(df['TDorSLD'] == 'TD') & (df['BIO_label'].str.startswith('B-'))])
        td_pct = (td_count / total_tokens) * 100 if total_tokens > 0 else 0
        stats.append({"Category": "High-Level: TD", "Description": "Typical Disfluencies", "Count": td_count, "Percentage": f"{td_pct:.2f}%"})
        print(f"{'Typical Disfluencies (TD)':<37}: {td_count:<6} ({td_pct:.2f}%)")

        sld_count = len(df[(df['TDorSLD'] == 'SLD') & (df['BIO_label'].str.startswith('B-'))])
        sld_pct = (sld_count / total_tokens) * 100 if total_tokens > 0 else 0
        stats.append({"Category": "High-Level: SLD", "Description": "Stuttering-Like Disfluencies", "Count": sld_count, "Percentage": f"{sld_pct:.2f}%"})
        print(f"{'Stuttering-Like Disfluencies (SLD)':<37}: {sld_count:<6} ({sld_pct:.2f}%)")
        
        print("="*40)
        pd.DataFrame(stats).to_csv("dataset_stats.csv", index=False)
        print("Statistics saved to: dataset_stats.csv")

if __name__ == "__main__":
    DATASET_PATH = "/Users/wanlingyeo/URECA/chat_transcripts"
    preprocessor = SALTPreprocessor()
    preprocessor.process_all(DATASET_PATH)
