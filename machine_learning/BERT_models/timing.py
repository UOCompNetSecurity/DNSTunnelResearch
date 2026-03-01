import sys
import platform
import time
import numpy as np
from csv_generator import generate_malicious_query, list_of_safe_queries
from model import *

def is_available(device_name:str) -> bool:
    devices = ["cuda", "mps", "cpu"]
    if device_name not in devices:
        raise ValueError("Invalid device name")

    if torch.cuda.is_available() and device_name == "cuda":
        return True
    elif torch.mps.is_available() and device_name == "mps":
        return True
    elif torch.cpu.is_available() and device_name == "cpu":
        return True
    return False

def check_devices(model: AutoModelForSequenceClassification) -> None:
    device = next(model.parameters()).device
    if device == "cpu":
        print(f"CPU: {platform.processor()}")
        print(f"CPU cores: {os.cpu_count()}")
    elif device == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

def anomoly_count(data, threshold:int) -> int:
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)

    # Calculate Z-scores for each data point
    z_scores = (data - mean) / std

    # Find indices of anomalies
    anomalies_indices = np.where(np.abs(z_scores) > threshold)[0]

    #Return number of anomolies
    return len(anomalies_indices)

def timing(model:AutoModelForSequenceClassification, safe_file:str, device:str) -> tuple[list[float]]:
    dangerous_queries = [generate_malicious_query() for i in range(100)]
    safe_queries =  list_of_safe_queries(safe_file, 100) 

    dangerous_pred_times = []
    safe_pred_times = []

    predict_float(dangerous_queries[0], model, device) #Warm up run

    for i in range(100):
        start_time = time.time()
        temp = predict_float(dangerous_queries[i], model, device)
        end_time = time.time()

        time_elapsed = end_time - start_time
        dangerous_pred_times.append(time_elapsed)

        start_time = time.time()
        temp = predict_float(safe_queries[i], model, device)
        end_time = time.time()

        time_elapsed = end_time - start_time
        safe_pred_times.append(time_elapsed)

    return dangerous_pred_times, safe_pred_times
    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:")
        print("  python3 timing.py <device> <model> <safe_query_file>")
        sys.exit(1)
    d_name = sys.argv[1]
    pretraining = sys.argv[2]
    safe_queries = sys.argv[3]

    if not is_available(d_name):
        print("Specified device not found")
        sys.exit(1)

    d_times, s_times = timing(d_name, pretraining, safe_queries)
    d_times_anom = anomoly_count(np.array(d_times), 5)
    s_times_anom  = anomoly_count(np.array(s_times), 5)

    if d_times_anom > 3 or s_times_anom > 3:
        print("Anomalous data detected terminating")
        sys.exit(1)
    
    print(f"Running predictions on {d_name}")
    print(f"Device info ...")
    check_devices(pretraining)
    print(f"Mean prediciton time for dangerous queries: {np.mean(d_times)}")
    print(f"Mean prediciton time for safe queries: {np.mean(s_times)}")