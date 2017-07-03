# -*- coding: utf-8 -*-
"""

@author: zhaojie


"""

import numpy as np
import scipy.signal as signal
from scipy.signal import hilbert
import multiprocessing


## 24 hour dentrending function
def dtrndanl(g):
    
    lg=g.shape[0]
    window=48
    res=np.zeros((g.shape[1],g.shape[0]))
    
    for k in range (res.shape[0]):
        id_g=g[:,k].T
        d_g=np.zeros((g.shape[0]-window+1,lg))
        s=np.zeros((g.shape[0]-window+1,lg))
        for j in range (g.shape[0]-window+1):
            d_g[j,j:j+window]=signal.detrend(id_g[j:j+window])
            s[j,j:j+window]=1
        for i in range(d_g.shape[1]):
            a=d_g[:,i]
            ss=s[:,i]
            ass=np.dot(a,ss)
            res[k,i]=ass/np.sum(ss)
           
    for i in range(res.shape[0]):
        res[i,:]=res[i,:]-np.mean(res[i,:])  
    return res


## continuous hilbert phase function
def HilbertPhase(res):
    analytic_signal = hilbert(res)
    phi = np.unwrap(np.angle(analytic_signal))
    return phi


## phase locking value
def phase_locking_value(theta1, theta2):
    complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
    plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
    return plv

## pairwise phase locking vlaue
def pairwise_plv(k,day,phi):
    l=np.shape(phi)[0]
    r=np.zeros((l,day))
    for n in range (l):
        for t in range (day):
            r[n,t]=phase_locking_value(phi[k,t*48:t*48+48],phi[n,t*48:t*48+48])
    return r


## multiprocessing pairwise phase locking vlaue
def PLV(phi,day):
    l=np.shape(phi)[0]
    cell_number=range(l)
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)
    results = [pool.apply(pairwise_plv, args=(k,day,phi)) for k in cell_number]
    #result = [p.get() for p in results]
    return results




                