"""Thin service entry; native backend/probe/RPC stay frozen v3.3."""
from preparations.idea1_v3_3 import server
from .policy import Policy

server.Policy=Policy
if __name__=='__main__':server.main()
