"""
Resolver -> Detector Event for each DNS query
"""

from dataclasses import dataclass
from typing import Optional 
from enum import Enum


# Only allowed types. 
class DNSQType(Enum): 
    A = 1      #Domain to IPV4
    AAAA = 28  #DOmain to IPV6
    NS = 2     #Specify Nameserver
    SOA = 6    #Authority 
    MX = 15    #Mail
    PTR = 12   #Reverse DNS 
    CNAME = 5  #Canonical name 


@dataclass(frozen=True)
class DNSQueryEvent:
    timestamp: float
    client_ip: str
    qname: str
    qtype: DNSQType
    query_size: int
    edns_size: Optional[int]


