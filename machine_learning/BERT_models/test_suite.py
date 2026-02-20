import sys
import torch
import pandas as pd
from model import *

DEFAULT_MALICIOUS = [
    "q+Z+/X8VBA.hidemyself.org",
    "q+Z8HXw1BA.hidemyself.org",
    "q+Z+i36jBA.hidemyself.org",
    "q+Z+mX6xBA.hidemyself.org",
    "q+Z+o367BA.hidemyself.org",
    "q+Z+uH7QBA.hidemyself.org",
    "q+Z+x37fBA.hidemyself.org",
    "q+Z+3X71BA.hidemyself.org",
    "q+Z+838LBA.hidemyself.org",
    "q+Z++38TBA.hidemyself.org"
]

DEFAULT_SAFE = [
    "weibo.cn",
    "abc.net.au",
    "ebay.be",
    "huffingtonpost.ca",
    "adschoom.com",
    "247wallst.com",
    "autoblog.com",
    "spiegel.de",
    "ic-live.com",
    "zenmate.com"
]


def main(data_path: str, query_column: str, label_column: str, m_indicator: str, threshold: float):
    print("Initializing Model ...")
    model_name = "amahdaouy/DomURLs_BERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        device = "cuda" 
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    model.eval() # Set model to evaluation mode
    print(f"Finished Initializing Model on {device}...")

    # Load the dataset
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    iter = 0

    print(f"Analyzing {len(df)} queries...")

    for index, row in df.iterrows():
        full_query = str(row[query_column])
        actual_label = str(row[label_column])

        # 1. Extract Subdomain (Split by dots and take everything before the domain.tld)
        prediction = analyze_subdomain_with_bert(full_query, tokenizer, model)

        # 4. Compare with ground truth
        # We assume 1 is Malicious and 0 is Benign
        is_malicious_pred = (prediction > threshold)
        is_malicious_actual = (actual_label == m_indicator)

        if is_malicious_pred and is_malicious_actual:
            true_positives += 1
        elif is_malicious_pred and not is_malicious_actual:
            false_positives += 1
        elif not is_malicious_pred and not is_malicious_actual:
            true_negatives += 1
        elif not is_malicious_pred and is_malicious_actual:
            false_negatives += 1

        iter += 1
        if(iter % 100 == 0):
            print(iter)

    # 5. Report Results
    total = len(df)
    accuracy = (true_positives + true_negatives) / total
    print("\n--- Detection Results ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"True Positives (Correct Detections): {true_positives}")
    print(f"True Negatives (Correct Benign): {true_negatives}")

def default_tests():
    model_name = "amahdaouy/DomURLs_BERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        device = "cuda" 
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    model.eval()

    for i in range(len(DEFAULT_MALICIOUS)):
        malcious_val = analyze_subdomain_with_bert(DEFAULT_MALICIOUS[i], tokenizer, model)
        safe_val = analyze_subdomain_with_bert(DEFAULT_SAFE[i], tokenizer, model)
        print(f"Malicious query {DEFAULT_MALICIOUS[i]} model prediction {malcious_val:.2%}")
        print(f"Safe query {DEFAULT_SAFE[i]} model prediction {safe_val:.2%}")
    
    return

if __name__ == "__main__":
    # Example usage: python script.py data.csv qname label "malicious"
    if len(sys.argv) == 6:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))
    elif len(sys.argv) ==  1:
        default_tests()
    else:
        print("Usage: python3 test_suite.py <csv_path> <query_col> <label_col> <malicious_indicator> <threshold>")
        print("Or no arguments for default tests")


    

    
    



