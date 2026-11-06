"""Exact native Euler node receipt contract for this D5/D10-only cohort."""
import math


def scheduled_nfe(nodes):
    if not isinstance(nodes,list) or len(nodes) not in (6,11):raise ValueError('Camera paths require an exact D5 or D10 time-node list')
    n=len(nodes)-1;expected=[1-i/n for i in range(n+1)]
    if any(type(x) not in (int,float) or not math.isfinite(x) or x!=y for x,y in zip(nodes,expected)):
        raise ValueError('Camera native Euler nodes differ from the exact frozen schedule')
    return n
