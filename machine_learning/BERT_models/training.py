import sys
from model import *

def train(input_file: str, output_file: str, arg_type: str = "Default") -> None:
    training, val = create_dataset(input_file)
    train_and_save(training, val, output_file, arg_type)

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage:")
        print("  python3 training.py <input_file> <output_file> [arg_type]")
        print("arg_type options: Default, Fast, HighAccuracy, LowMemory, Regularized")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2]
    arg_type    = sys.argv[3] if len(sys.argv) == 4 else "Default"
    train(input_file, output_file, arg_type)