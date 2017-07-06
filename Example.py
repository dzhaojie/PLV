# -*- coding: utf-8 -*-
"""

@author: zhaojie


"""


import scipy.io as sio
import PhaseVariance as PV
import numpy as np





if __name__ == '__main__':

    Num_of_Cell=sio.loadmat('C:/Users/username/Desktop/Num_all_cell_list.mat')
    NoC=Num_of_Cell['NoC']
    RhodamineB=sio.loadmat('C:/Users/username/Desktop/light_intensity_calculated_from_RhodamineB.mat')
    p=RhodamineB['p']
    
    for w in range(1,NoC.shape[1]):
     
        num_of_cell=NoC[w,0]
        results=np.empty((0,11)) ## size is decided by the length of the measured time series data
        for field_of_view in range(1,100):
            try:
              Single_Cell=sio.loadmat('C:/Users/username/Desktop/'+str(field_of_view)+'/'+str(num_of_cell)+'cell/g.mat')
              Good_Bad_Cell_Indicator=sio.loadmat('C:/Users/username/Desktop/'+str(field_of_view)+'/'+str(num_of_cell)+'cell/gb.mat')
              Droplet_ID=sio.loadmat('C:/Users/username/Desktop/'+str(field_of_view)+'/'+str(num_of_cell)+'cell/droplet_id.mat')  
              Cell_X_Coordinates=sio.loadmat('C:/Users/username/Desktop/'+str(field_of_view)+'/'+str(num_of_cell)+'cell/X.mat')
              Cell_Y_Coordinates=sio.loadmat('C:/Users/username/Desktop/'+str(field_of_view)+'/'+str(num_of_cell)+'cell/Y.mat')
            except:
                continue
            
            gn=Single_Cell['g']
            g=gn/p.T
            gb=Good_Bad_Cell_Indicator['gb']
            DI=Droplet_ID['DI']
            X=Cell_X_Coordinates['X']
            Y=Cell_Y_Coordinates['Y']
        
            unique_DI=np.unique(DI)
            for di in range(unique_DI.shape[0]):
                col=np.where(DI == unique_DI[di])[1]
                gg=np.empty((g.shape[0], 0))
                c=0
                ccol=[]            
                for k in range(col.shape[0]):
                    if gb[0,col[k]]==1:
                       gtemp=np.reshape(g[:,col[k]],(g.shape[0],1))
                       ccol.append(col[k])
                       gg=np.append(gg,gtemp,axis=1)
                    for i in range(col.shape[0]):   
                        cell_cell_dis=np.sqrt(np.square(X[0,col[k]]-X[0,col[i]])+np.square(Y[0,col[k]]-Y[0,col[i]]))
                        if cell_cell_dis>0 and cell_cell_dis<=6.7:## define the distance that is considered for cell connecting with each other
                            c=c+1
                
                result=np.empty((0,11))  ## size is decided by the length of the measured time series data
                if gg.shape[1]>1:
                    res=PV.dtrndanl(gg)
                    phi=PV.HilbertPhase(res)
                    day=10
                    l=np.shape(phi)[0]
                    output=PV.PLV(phi,day)
                    R=np.ones((l,day))
                    for i in range(l):
                        a=np.ma.array(output[i], mask=False)
                        a.mask[i,:]=True
                        R[i,:]=a.mean(axis=0)
                    R_mean=np.mean(R,axis=0)
                    result=np.append(R_mean,c/2)
                    result=result.reshape((1,11))  ## size is decided by the length of the measured time series data
                   
                    
                results=np.append(results,result,axis=0)
        np.save('C:/Users/username/Desktop/'+str(w)+'cell.npy',results)           
                    
            
           
            
            
           
            
            
            