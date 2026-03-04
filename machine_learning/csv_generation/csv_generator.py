import sys
import base64
import pandas as pd
from wonderwords import RandomWord
import random as rand
import string

DOMAINS = [".net", ".is", ".com", ".edu", ".co", ".org", ".ru", ".me", ".site", ".de"]

def random_characters(num_chars: int) -> str: 
    chars = ""
    for _ in range(num_chars): 
        chars += rand.choice(string.ascii_letters)
    return chars

def generate_malicious_query(generator: RandomWord) -> str:
    text = generator.word()
    attacker_domain = f"{generator.word()}{DOMAINS[rand.randint(0, 9)]}"
    encoded_text = base64.urlsafe_b64encode(text.encode()).decode().strip("=")

    return f"{encoded_text}.{random_characters(3)}.{attacker_domain}"

def list_of_safe_queries(file_path:str, num_queries = 1000) -> list[str]:
    query_list = []

    with open(file_path, 'r') as file:
        lines_list = file.read().splitlines()
        num_lines = len(lines_list)

        for i in range(num_queries):
            idx = rand.randint(0, num_lines - 1)
            query_list.append(lines_list[idx])
            
    return query_list

def create_csv(num_lines: int, safe_query_list:list[str], csv_path:str) -> None:
    data = {"Value": [], "Query":[]}
    safe_query_len = len(safe_query_list)
    gen = RandomWord()

    for i in range(num_lines):
        is_malicious = bool(rand.getrandbits(1))
        if is_malicious:
            data["Query"].append(generate_malicious_query(gen))
            data["Value"].append(1)
        else:
            data["Query"].append(safe_query_list[rand.randint(0, safe_query_len - 1)])
            data["Value"].append(0)
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    print("Successfully wrote to CSV")

    return

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage:")
        print("  python3 csv_generator.py <safe_query_file> <output_csv> <num lines> [safe_query_list_length]")
        sys.exit(1)
    else:
        if len(sys.argv) == 4:
            q_list = list_of_safe_queries(sys.argv[1])
        else:
            q_list = list_of_safe_queries(sys.argv[1], int(sys.argv[3]))
        create_csv(int(sys.argv[3]), q_list, sys.argv[2])



        


    



