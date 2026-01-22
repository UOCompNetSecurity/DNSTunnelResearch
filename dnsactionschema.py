"""
Detector -> Resolver Control Message whenever an action is determined on a domain through classification/analysis
"""

from dataclasses import dataclass
from enum import Enum

class ActionType (str, Enum):
    """
    Immediately deny the DNS query and return a failure response, without attempting DNS resolution
    Use Case: High confidence in a domain or pattern being malicious 
    """
    BLOCK = "BLOCK"

    """
    Resolve the DNS query to a non-malicious destination (sinkhole IP), not the real address
    Use Case: Containment and monitoring activity
    """
    SINKHOLE = "SINKHOLE"

    """
    Explicitly allow the DNS query to resolve normally, overriding the fast path analysis decision
    Use Case: White-list known-safe domains and supress false positives
    """
    ALLOW = "ALLOW"

    """
    Throttle DNS queries by limiting the number of requests over a certain time period. 
    Use Case: Behavior is suspicious but confidence in malicious activity is not high enough for a block action
    """
    RATE_LIMIT = "RATE_LIMIT"

class ScopeType (str, Enum):
    EXACT = "exact"
    SUFFIX = "suffix"
    WILDCARD = "wildcard"

@dataclass
class EnforcementAction:
    action: ActionType
    scope: ScopeType
    value: str
    ttl: int
    confidence: float