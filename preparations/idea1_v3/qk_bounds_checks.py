"""CPU-only address and valid-index algebra check for the exact failed shape."""
import numpy as np


def main():
    m,n,k=408,571,256
    assert (m+31)//32*32-m==8 and (n+31)//32*32-n==5
    rng=np.random.default_rng(1701)
    q=rng.integers(-2,3,(m,k),dtype=np.int32); keys=rng.integers(-2,3,(n,k),dtype=np.int32)
    result=np.zeros((m,n),np.int32)
    for i in range(0,m,32):
        for j in range(0,n,32):
            rows=i+np.arange(32); columns=j+np.arange(32)
            acc=np.zeros((32,32),np.int32)
            for start in range(0,k,64):
                features=start+np.arange(64)
                qm=(rows[:,None]<m)&(features[None,:]<k)
                km=(columns[:,None]<n)&(features[None,:]<k)
                qa=np.zeros((32,64),np.int32); ka=qa.copy()
                qr,qd=np.nonzero(qm); kr,kd=np.nonzero(km)
                assert np.all(rows[qr]*k+features[qd]<m*k)
                assert np.all(columns[kr]*k+features[kd]<n*k)
                qa[qr,qd]=q[rows[qr],features[qd]]; ka[kr,kd]=keys[columns[kr],features[kd]]
                acc+=qa@ka.T
            mr,nr=min(32,m-i),min(32,n-j)
            assert (i+mr-1)*n+j+nr-1<m*n
            result[i:i+mr,j:j+nr]=acc[:mr,:nr]
    np.testing.assert_array_equal(result,q@keys.T)
    print('QK failed shape M408/N571/K256: legacy Q8/K5 unmasked extra rows; all new load/store addresses bounded; valid-index tiled dot algebra exact: OK')

if __name__=='__main__': main()
