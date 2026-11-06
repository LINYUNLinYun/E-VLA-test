"""Fresh supplement service; original RPC transport reused without source edits."""
def main():
    from preparations.idea1_v3 import pi05,branch_server
    from preparations.idea1_v3_supplement.pi05 import Pi05Policy
    pi05.Pi05Policy=Pi05Policy
    branch_server.main()

if __name__=='__main__': main()
