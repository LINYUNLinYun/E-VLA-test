"""Prospective exact Euler receipts, D2/D10 only."""
import math


def scheduled_nfe(nodes):
    if not isinstance(nodes,list) or len(nodes) not in (3,11):raise ValueError('Future cohort requires exact D2/D10 Euler nodes')
    n=len(nodes)-1
    if any(type(v) not in (int,float) or not math.isfinite(v) or v!=1-i/n for i,v in enumerate(nodes)):
        raise ValueError('Exact adapter Euler nodes changed')
    return n
