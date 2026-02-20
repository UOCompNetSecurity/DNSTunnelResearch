from typing import Optional, List
from dataclasses import dataclass
import ipaddress
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import math
from collections import Counter
from ...dnseventschema import *

#======================================================================
# Helper Functions
#======================================================================
def determine_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def calculate_length(text: str) -> int:
    return len(text) if isinstance(text, str) else 0

def calculate_entropy(text: str) -> float:
    if not text or not isinstance(text, str): 
        return 0.0
    probabilities = [n / len(text) for n in Counter(text).values()]
    return -sum(p * math.log2(p) for p in probabilities)


#======================================================================
# Error Handling
#======================================================================
class CSVReadError(Exception):
    """Base class for exceptions in this project."""
    def __init__(self, message):
        super().__init__(message)

class MappingError(Exception):
    """Base class for exceptions in this project."""
    def __init__(self, message):
        super().__init__(message)

#======================================================================
# Detector Classes
#======================================================================
class DNSTunnelingMLP(nn.Module):
    def __init__(self, input_size):
        super(DNSTunnelingMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

#======================================================================
# CSV conversion
#======================================================================
class QueryData:
    def __init__(self):
        self.timestamps: Optional[List[float]] = None
        self.client_ips: Optional[List[int]] = None
        self.rrtypes: Optional[List[str]] = None
        self.protocols: Optional[List[int]] = None
        self.query_names: List[str] = []
        self.query_sizes: List[int] = []
        self.max_query_size: int = 0
        self.q_entropy_vals: List[float] = []
        self.labels: List[int] = []

    def find_data(self, request: str) -> list:
        mapping = {
            "timestamps": self.timestamps,
            "client ips": self.client_ips,
            "rr types": self.rrtypes,
            "protocols": self.protocols,
            "query names": self.query_names,
            "query sizes": self.query_sizes,
            "entropy values": self.q_entropy_vals
        }
        if request in mapping:
            return mapping[request]
        raise ValueError(f"Invalid request: {request}")
    
    def determine_longest_query(self):
        if len(self.query_sizes) == 0:
            return 0
        else:
            self.max_query_size = max(self.query_sizes)

@dataclass
class DNSColumnMapping:
    """
    Standardized object for mapping CSV headers to DNS features.
    Defaults to 'N/A' so you only have to specify what you need.
    """
    queries: str = "N/A"      # The primary DNS query string
    values: str = "N/A"           # The label (0 or 1)
    timestamps: str = "N/A"
    client_ips: str = "N/A"
    rrtypes: str = "N/A"
    protocols: str = "N/A"

def create_column_mapping(mapping: list[tuple[str, str]]) -> DNSColumnMapping:
    mapping_object = DNSColumnMapping()
    
    # Map the "friendly" string names to the actual class attribute names
    # This acts as a lookup table to avoid messy if/else blocks
    key_map = {
        "queries": "queries",
        "values": "values",
        "timestamps": "timestamps",
        "client ips": "client_ips", # Notice the space to underscore conversion
        "rr types": "rrtypes",
        "protocols": "protocols"
    }

    for key_name, csv_header in mapping:
        if key_name in key_map:
            # Get the correct attribute name (e.g., "client_ips")
            attribute = key_map[key_name]
            # Set the value on the object
            setattr(mapping_object, attribute, csv_header)
        else:
            raise MappingError(f'DNSColumnMapping is not compatible with: {key_name}')
            
    return mapping_object

def load_mapping_from_file(file_path: str) -> DNSColumnMapping:
    """
    Reads a .txt file where each line is: 'characteristic header'
    Example line: 'queries domain_name_column'
    """
    pairs = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # 1. Clean the line and skip empty lines or comments
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue
                
                # 2. Split the line into exactly two parts
                # maxsplit=1 ensures that if a column header has a space, 
                # it doesn't break the logic.
                parts = clean_line.split(maxsplit=1)
                
                if len(parts) == 2:
                    # Append the tuple (characteristic, column_header)
                    pairs.append((parts[0], parts[1]))
                else:
                    print(f"Warning: Skipping malformed line: '{line.strip()}'")

        # 3. Use your existing creation function to return the object
        return create_column_mapping(pairs)

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
            
def convert_csv(filepath: str, mapping: DNSColumnMapping, v4_or_v6=-1) -> QueryData:
    df = pd.read_csv(filepath)
    q_data = QueryData()
    
    # 1. Process Queries (Mandatory)
    if mapping.queries == "N/A" or mapping.queries not in df.columns:
        raise CSVReadError("CSV contains no queries or column missing")
    
    q_data.query_names = df[mapping.queries].astype(str).tolist()
    
    # 2. Calculate Features (Entropy & Size)
    for name in q_data.query_names:
        q_data.query_sizes.append(calculate_length(name))
        q_data.q_entropy_vals.append(calculate_entropy(name))

    # 3. Process Labels (Values)
    if mapping.values != "N/A":
        q_data.labels = df[mapping.values].astype(int).tolist()

    # 4. Process Optional Columns
    if mapping.timestamps != "N/A":
        q_data.timestamps = df[mapping.timestamps].astype(float).tolist()

    if mapping.protocols != "N/A":
        proto_map = {"UDP": 1, "TCP": 0}
        raw_protos = df[mapping.protocols].astype(str).tolist()
        q_data.protocols = [proto_map.get(p.upper(), -1) for p in raw_protos]

    if mapping.client_ips != "N/A":
        q_data.client_ips = []
        for ip in df[mapping.client_ips]:
            addr = ipaddress.ip_address(ip)
            q_data.client_ips.append(int(addr))

    return q_data

#======================================================================
# Tensor Construction (Consolidated)
#======================================================================
def construct_dataset_tensors(data: QueryData, variables: List[str], device: torch.device) -> tuple[torch.tensor]:
    # Dynamically gather all requested features
    feature_lists = [data.find_data(var) for var in variables]
    
    # zip(*feature_lists) creates tuples of features per row
    features = torch.tensor(list(zip(*feature_lists)), dtype=torch.float32).to(device)
    labels = torch.tensor(data.labels, dtype=torch.float32).to(device).view(0, 1)

    return features, labels

#======================================================================
# DNSQueryEvent Object Conversion
#======================================================================
@dataclass
class DNSModelFeatureEvent(DNSQueryEvent):
    # These now act as "Toggle Switches" you can set during initialization
    use_entropy: bool = True
    use_length: bool = True
    use_timestamp: bool = False
    use_client_ip: bool = False
    use_protocol: bool = False

    @classmethod
    def from_parent(cls, parent: DNSQueryEvent, **usage_flags):
        """Creates a child instance from a parent instance plus specific flags."""
        return cls(**parent.__dict__, **usage_flags)

    def get_feature_vector(ptr) -> list:
        """Extracts values only for features marked as True."""
        features = []
        if ptr.use_entropy:
            features.append(calculate_entropy(ptr.query_name))
        if ptr.use_length:
            features.append(calculate_length(ptr.query_name))
        if ptr.use_timestamp:
            features.append(float(ptr.timestamp))
        if ptr.use_client_ip:
            features.append(float(int(ipaddress.ip_address(ptr.client_ip))))
        if ptr.use_protocol:
            # 1 for UDP, 0 for TCP as defined in your logic
            proto_val = 1 if str(ptr.protocol).upper() == "UDP" else 0
            features.append(float(proto_val))
        return features

#======================================================================
# Training and Evaluation
#======================================================================
def train_dns_model_mlp(model, feature_tns, label_tns, epochs=50) -> None:
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(feature_tns)
        loss = criterion(outputs, label_tns)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def evaluate_query(detector: DNSTunnelingMLP, query: DNSQueryEvent, device: str, 
                   use_timestamp=False, use_client_ip=False, use_protocol=False) -> float:
    # 1. Upgrade to the feature-aware object using explicit keyword arguments
    feature_event = DNSModelFeatureEvent.from_parent(
        query, 
        use_timestamp=use_timestamp, 
        use_client_ip=use_client_ip, 
        use_protocol=use_protocol
    )
    
    # 2. Extract the vector (Make sure this returns a list of floats)
    features = feature_event.get_feature_vector() 
    
    # 3. Convert to Tensor and ensure the shape is [Batch, Features]
    # If features is [0.5, 12], this makes it [[0.5, 12]]
    input_tensor = torch.tensor([features], dtype=torch.float32).to(device)
    
    # 4. Perform Inference
    detector.eval()
    with torch.no_grad():
        prediction = detector(input_tensor)
    
    return prediction.item()


    
    

    

    
    

    

    
    
    
