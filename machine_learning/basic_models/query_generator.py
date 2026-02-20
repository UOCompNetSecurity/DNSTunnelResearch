import time
import random
import csv
from ...dnseventschema import *
from machine_learning_detection import DNSColumnMapping

def generate_queries(query_csv_path: str, column_mapping: DNSColumnMapping) -> list[DNSQueryEvent]:
    query_list = []
    default_client_ip = "127.0.0.1"
    default_rrtype = DNSRRType.A
    default_protocol = DNSProtocol.TCP
    query_file = csv.reader(query_csv_path)
    start_time = time.perf_counter()
    for line in query_file:
        query = DNSQueryEvent()
        if column_mapping.client_ips != "N/A":
            query.client_ip = line[column_mapping.client_ips]
        else:
            query.client_ip = default_client_ip
        if column_mapping.rrtypes != "N/A":
            query.rrtype = line[column_mapping.rrtypes]
        else:
            query.client_ip = default_rrtype
        if column_mapping.protocols != "N/A":
            query.protocol = line[column_mapping.rrtypes]
        else:
            query.protocol = default_protocol
        query.qname = column_mapping.queries
        query.query_size = len(query.qname)
        query.timestamp = time.perf_counter() - start_time
        query_list.append(query)
    return query_list


        


