"""Dedicated fresh D2 model process using the existing RPC transport."""
def main():
    from preparations.idea1_v3 import pi05,branch_server
    from preparations.idea1_pi05_d2_search.policy import Pi05Policy
    pi05.Pi05Policy=Pi05Policy
    branch_server.main()

if __name__=='__main__': main()
