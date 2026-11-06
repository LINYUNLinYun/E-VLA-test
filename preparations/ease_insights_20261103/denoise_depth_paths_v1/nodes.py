"""Prospective exact Euler receipts, including D3/D4 absent from the old helper."""
import math


def scheduled_nfe(nodes):
    if not isinstance(nodes,list) or len(nodes) not in (3,4,5,11):raise ValueError('Future cohort requires exact D2/D3/D4/D10 Euler nodes')
    n=len(nodes)-1
    if any(type(v) not in (int,float) or not math.isfinite(v) or v!=1-i/n for i,v in enumerate(nodes)):
        raise ValueError('Exact adapter Euler nodes changed')
    return n
