import sys
from machine_learning_detection import *
from query_generator import generate_queries 

def main(training_data_path: str, simulation_data_path: str, training_mapping_path:str, 
simulation_mapping_path:str, parameter_list: list[str]) -> None:
    training_mapping = load_mapping_from_file(training_mapping_path)
    simulation_mapping = load_mapping_from_file(simulation_mapping_path)
    torch_device = determine_device()

    print("Generating Detection Model ...")
    training_data = convert_csv(training_data_path, training_mapping)
    data_size = len(training_data.query_names)
    model = DNSTunnelingMLP(data_size)
    feature_tensor, label_tensor = construct_dataset_tensors(training_data, parameter_list, torch_device)
    train_dns_model_mlp(model, feature_tensor, label_tensor)
    print("Finished Training Model")

    print("Generating Queries ...")
    queries = generate_queries(simulation_data_path, simulation_mapping)
    query_count = len(queries)
    print("Finished generating Queries")

    print(f"running simulation, evaluating {query_count} queires")
    malicious_queries = 0
    detection_threshold = 0.7
    extra_features = [False, False, False] #Do we use timestamps, client ip, or protocol in our model
    if "timestamp" in parameter_list:
        extra_features[0] = True
    if "client ip" in parameter_list:
        extra_features[1] = True
    if "protocol" in parameter_list:
        extra_features[2] = True
    for query in queries:
        prediction = 0.0
        if not (extra_features[0] and extra_features[1] and extra_features[2]):
            prediction = evaluate_query(model, query, torch_device)
        else:
            prediction = evaluate_query(model, query, torch_device, extra_features[0], extra_features[1], extra_features[2])

        if prediction > detection_threshold:
            malicious_queries += 1

    print(f"simulation finished: {malicious_queries} out of {query_count} were determined to be malicious")

if __name__ == "__main__":
    if len(sys.argv) > 6 and len(sys.argv) < 12:
        parameter_list = []
        for i in range(6, len(sys.argv)):
            parameter_list.append(sys.argv[i])
        main(sys.argv[2], sys.orig_argv[3], sys.argv[4], sys.argv[5], parameter_list)
    else:
        print("Usage: python3 detection_simulation.py <path to training data> <path to simulation data> <training data mapping>"
        "<simulation data mapping> <Up to five model parameters>")