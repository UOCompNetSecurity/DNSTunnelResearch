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

def print_stats(label: str, preds: list, threshold: float) -> None:
    if not preds:
        print(f"  No {label} queries found.")
        return
    arr = np.array(preds)
    above = int(np.sum(arr > threshold))
    below = len(arr) - above
    print(f"  Count:              {len(arr)}")
    print(f"  Mean:               {arr.mean():.4f}")
    print(f"  Std Dev:            {arr.std():.4f}")
    print(f"  Min:                {arr.min():.4f}")
    print(f"  Max:                {arr.max():.4f}")
    print(f"  Median:             {np.median(arr):.4f}")
    print(f"  Above Threshold:    {above} ({100 * above / len(arr):.1f}%)")
    print(f"  Below Threshold:    {below} ({100 * below / len(arr):.1f}%)")

def predict_default(pretraining_file: str) -> None:
    temp = 1.0 #default temperature of 1
    hardware = determine_device()

    model = AutoModelForSequenceClassification.from_pretrained(pretraining_file)
    model.to(hardware)
    model.eval()

    for q in DEFAULT_MALICIOUS:
        pred = predict_float(q, model, temp, hardware)
        print(f"Malicious Query: {q}, Prediction: {pred}")

    for q in DEFAULT_SAFE:
        pred = predict_float(q, model, temp, hardware)
        print(f"Non Malicious Query: {q}, Prediction: {pred}")

def predict_queryfile(pretraining_file: str, input_file: str, temperature: float, threshold: float, iterations: int = 1000) -> None:
    hardware = determine_device()

    model = AutoModelForSequenceClassification.from_pretrained(pretraining_file)
    model.to(hardware)
    model.eval()

    df = pd.read_csv(input_file)
    query_list = df['Query'].tolist()
    val_list = df['Value'].tolist()
    iterations = min(iterations, len(query_list))

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(iterations):
        predicted_malicious = predict_float(query_list[i], model, temperature, hardware) > threshold
        is_malicious = val_list[i] == 1
        if predicted_malicious and is_malicious:
            true_positives += 1
        elif predicted_malicious and not is_malicious:
            false_positives += 1
        elif not is_malicious:
            true_negatives += 1
        else:
            false_negatives += 1

    print("Final Results:")
    print(f"Total Queries Analyzed: {iterations}")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")

def predict_queryfile_distribution(pretraining_file: str, input_file: str, temperature: float, threshold: float, iterations: int = 1000) -> None:
    hardware = determine_device()

    model = AutoModelForSequenceClassification.from_pretrained(pretraining_file)
    model.to(hardware)
    model.eval()

    df = pd.read_csv(input_file)
    query_list = df['Query'].tolist()
    val_list = df['Value'].tolist()
    iterations = min(iterations, len(query_list))

    malicious_predictions = []
    benign_predictions = []

    for i in range(iterations):
        predicted_value = predict_float(query_list[i], model, temperature, hardware)
        is_malicious = val_list[i] == 1
        if is_malicious:
            malicious_predictions.append(predicted_value)
        else:
            benign_predictions.append(predicted_value)

    print(f"\n{'='*45}")
    print(f"  Distribution Report (threshold={threshold})")
    print(f"{'='*45}")
    print(f"\n  Malicious Queries:")
    print(f"  {'-'*40}")
    print_stats("malicious", malicious_predictions, threshold)
    print(f"\n  Benign Queries:")
    print(f"  {'-'*40}")
    print_stats("benign", benign_predictions, threshold)
    print(f"\n{'='*45}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 test_suite.py predict_default <pretraining_file>")
        print("  python3 test_suite.py predict_binary <pretraining_file> <input_file> <threshold> [iterations]")
        print("  python3 test_suite.py predict_distribution <pretraining_file> <input_file> <threshold> [iterations]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "predict_default":
        if len(sys.argv) != 3:
            print("Usage: python3 test_suite.py predict_default <pretraining_file>")
            sys.exit(1)
        predict_default(sys.argv[2])

    elif command == "predict_binary":
        if len(sys.argv) < 6 or len(sys.argv) > 7:
            print("Usage: python3 test_suite.py predict_binary <pretraining_file> <input_file> <prediction_temp> <threshold> [iterations]")
            sys.exit(1)
        pretraining_file = sys.argv[2]
        input_file = sys.argv[3]
        temp = float(sys.argv[4])
        threshold = float(sys.argv[5])
        iterations = int(sys.argv[6]) if len(sys.argv) == 7 else 1000
        predict_queryfile(pretraining_file, input_file, temp, threshold, iterations)

    elif command == "predict_distribution":
        if len(sys.argv) < 6 or len(sys.argv) > 7:
            print("Usage: python3 test_suite.py predict_distribution <pretraining_file> <input_file> <prediction_temp> <threshold> [iterations]")
            sys.exit(1)
        pretraining_file = sys.argv[2]
        input_file = sys.argv[3]
        temp = float(sys.argv[4])
        threshold = float(sys.argv[5])
        iterations = int(sys.argv[6]) if len(sys.argv) == 7 else 1000
        predict_queryfile_distribution(pretraining_file, input_file, temp, threshold, iterations)

    else:
        print(f"Unknown command: '{command}'")
        print("Valid commands: predict_default, predict_binary, predict_distribution")
        sys.exit(1)