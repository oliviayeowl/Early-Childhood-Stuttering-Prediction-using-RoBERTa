import os
import re
import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def get_pos_neighborhood(text, target_patterns, window=2):
    """
    Finds the POS tags of words surrounding disfluency markers.
    """
    
    placeholders = {}
    temp_text = text
    for i, (name, pattern) in enumerate(target_patterns.items()):
        placeholder = f" DISFLUENCY_MARKER_{i} "
        placeholders[placeholder.strip()] = name
        temp_text = re.sub(pattern, placeholder, temp_text)
    
    doc = nlp(temp_text)
    neighborhoods = []
    
    for i, token in enumerate(doc):
        if token.text in placeholders:
            label = placeholders[token.text]
            
            # Get preceding POS tags
            preceding = [doc[j].pos_ for j in range(max(0, i - window), i)]
            # Get following POS tags
            following = [doc[j].pos_ for j in range(i + 1, min(len(doc), i + window + 1))]
            
            neighborhoods.append({
                'type': label,
                'preceding': preceding,
                'following': following
            })
            
    return neighborhoods

def main():
    DATA_PATH = './training_data.jsonl' # Changed to JSONL
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Please ensure the data cleaning script generates this file.")
        return

    # Load data based on file extension
    if DATA_PATH.endswith('.csv'):
        df = pd.read_csv(DATA_PATH)
    elif DATA_PATH.endswith('.jsonl'):
        df = pd.read_json(DATA_PATH, lines=True)
    else:
        print("Error: Unsupported data file format. Please use .csv or .jsonl")
        return
    
    # Ensure required columns exist
    if 'original_utterance' not in df.columns or 'has_stuttering' not in df.columns:
        print("Error: 'original_utterance' or 'has_stuttering' column not found in the dataset.")
        return

    # Define SLD patterns (matching the cleaning script)
    sld_patterns = {
        'PW': r'\[\^\s*PW\s*(\d+)\]',
        'WW': r'\[\^\s*WW\s*(\d+)\]',
        'DP': r'\[\^\s*DP\s*\]',
        'P': r'\[\^\s*P\s*(\d+)\]'
    }
    
    print("Analyzing syntactic neighborhoods...")
    all_neighborhoods = []
    # Filter for utterances with stuttering and iterate through their original text
    for text in df[df['has_stuttering'] == 1]['original_utterance']:
        all_neighborhoods.extend(get_pos_neighborhood(text, sld_patterns))
    
    if not all_neighborhoods:
        print("No SLD neighborhoods found.")
        return

    # Convert to DataFrame for analysis
    neigh_df = pd.DataFrame(all_neighborhoods)
    
    # Flatten preceding and following for overall distribution
    pre_tags = [tag for sublist in neigh_df['preceding'] for tag in sublist]
    post_tags = [tag for sublist in neigh_df['following'] for tag in sublist]
    
    pre_counts = Counter(pre_tags)
    post_counts = Counter(post_tags)
    
    print("\nTop POS tags PRECEDING SLDs:")
    for tag, count in pre_counts.most_common(5):
        print(f"{tag}: {count}")
        
    print("\nTop POS tags FOLLOWING SLDs:")
    for tag, count in post_counts.most_common(5):
        print(f"{tag}: {count}")

    # Visualization: Preceding POS Tags
    plt.figure(figsize=(12, 6))
    pre_df = pd.DataFrame(pre_counts.most_common(10), columns=['POS Tag', 'Count'])
    sns.barplot(x='Count', y='POS Tag', data=pre_df, palette='viridis')
    plt.title('Top 10 POS Tags Preceding Stuttering-Like Disfluencies')
    plt.tight_layout()
    plt.savefig('./preceding_pos_distribution.png')
    
    # Visualization: Following POS Tags
    plt.figure(figsize=(12, 6))
    post_df = pd.DataFrame(post_counts.most_common(10), columns=['POS Tag', 'Count'])
    sns.barplot(x='Count', y='POS Tag', data=post_df, palette='magma')
    plt.title('Top 10 POS Tags Following Stuttering-Like Disfluencies')
    plt.tight_layout()
    plt.savefig('./following_pos_distribution.png')

    # Specific check for Pronouns and Conjunctions
    function_words = ['PRON', 'CCONJ', 'SCONJ', 'DET', 'ADP']
    print("\nFunction Word Presence in SLD Neighborhoods:")
    for fw in function_words:
        pre_fw = pre_counts.get(fw, 0)
        post_fw = post_counts.get(fw, 0)
        print(f"{fw}: Preceding={pre_fw}, Following={post_fw}")

if __name__ == "__main__":
    main()
