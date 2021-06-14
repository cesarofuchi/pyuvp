import pyuvp.ultrasound as usp
#from ultrasound import Ultrasound
from scipy.signal import lfilter, hilbert,medfilt2d,correlate
#import matplotlib.pyplot as plt
import numpy as np
import math
import time

class UltrasoundDoppler:
    

    def __init__(self,usdata,ns,nc,ovs,ovt,cycles=4):
        self.usdata=usdata
        self.ns=ns
        self.nc=nc
        self.ovs=ovs
        self.ovt=ovt
        
        ## Matched Filter
        # 1 create signal
        t=np.arange(0,cycles/usdata.fc,1/usdata.fs)
        impulse=np.sin(2*math.pi*usdata.fc*t)

        h=np.hamming(len(impulse))
        impulse_response=impulse*h.transpose()   

        #plt.plot(impulse_response)
        #plt.show()
       
        # axis=0 is fundamental
        # https://stackoverflow.com/questions/16936558/matlab-filter-not-compatible-with-python-lfilter

        filt_impulse=np.flipud(impulse_response)       
        #scipy.signal.lfilter(b, np.ones(1), x)
        filtered_data=lfilter(filt_impulse,np.ones(1),usdata.data,axis=0)

        # fig, ax=plt.subplots(figsize=(15, 9))
        # ax.set_titusData=np.flipud(filtered_data)
        # ax.set_ylabel('amplitude')
        # ax.set_xlabel('samples')
        # i=1
        # y1=usdata.data[:,i]
        # y2=filtered_data[:,i]
        # ax.plot(y1,'r',y2,'b',linewidth=0.5)
        # ax.legend(['Original','Matched filter'])    
        # plt.show()
        #         

        ## Hilbert transform to get IQ data
        # transformada de hilbert axis=0 is utterly important!
        analytic_h=hilbert(filtered_data,axis=0)
        tp=np.arange(0,len(analytic_h),1)
        samples,pulses=filtered_data.shape
        
        #criar vetor de tempo inicio=0 : fim= (amostras -1 )/fs : passo= 1/fs 
        t=np.arange(0,(samples)/usdata.fs,1/usdata.fs)
        #t=np.arange(0,samples,1)
        #t=t/usdata.fs
        t=np.kron(np.ones((pulses,1)),t)
        t=t.transpose()

        #IQ demodulation
        #decimation
        dec=1        
        #iq=h*exp(-1i*2*pi*(fc)*t);        
        iq=analytic_h*np.exp(-1j*2*math.pi*usdata.fc*t)

        self.iq=iq
        

    def auto_correlation(self,c,np_range=[-2,-1,0,1,2],debug=False,method='normal'):
       
        if method in ['normal']:
            acm=True
        elif method in ['extended']:
            acm=False
        else:
            raise ValueError("Method must be 'normal' or 'extended'."
                            % method)
        nc=self.nc
        ns=self.ns        
        ovs=self.ovs        
        ovt=self.ovt        
        fprf=self.usdata.fprf
        fc=self.usdata.fc
        fs=self.usdata.fs
        RMS=self.iq[:,5].std()       
        np_range=np.array(np_range)
        
        samples,pulses = self.iq.shape

        rest=np.remainder(samples*-ns,ns*ovs)
        if rest !=0:
            pad_n=int(ns*ovs-rest)
            pad=np.zeros(pad_n,pulses)
            self.iq= np.array([iq,pad])	
        
        nchannels=samples/(ovs*ns)
        nchannels=int(nchannels)
        nchannelst=np.floor((pulses-nc)/(ovt*nc))+1
        nchannelst=int(nchannelst)

        flow=np.zeros((nchannels,nchannelst))
        desv=np.zeros((nchannels,nchannelst))

        tt = time.time()
        # do stuff
        for i in range(nchannelst):    
            p_i=i*nc*ovt
            p_f=p_i+nc
            for j in range(nchannels):
                ps_i=(j)*ns*ovs
                ps_f=ps_i+ns
                dataDop=self.iq[ps_i:ps_f,p_i:p_f]
                u=dataDop.shape        
                Nsamples=u[0]
                M=u[1]
                #sum all data (is equivalent to process all lines)
                vdata=np.sum(dataDop,axis=0)                
                if np.abs(vdata.std()) > np.abs(RMS/500):
                    #optional
                    desv[j,i]=np.abs(vdata).std()                    
                    v0=vdata[1:M]               
                    v1=vdata[0:(M-1)]                                     
                    auto=v0.dot(v1.conj())
                    #auto1 = np.vdot(vdata[0:(M-1)],vdata[1:M])                    
                    #auto = v0@v1.conj().transpose()                                
                    phi_est =  np.angle(auto)                   
                else:
                    phi_est=0
                
                phi_auto=phi_est

                ############
                #EAM        
                if acm == False:
                    phi_possible=phi_auto+np_range*2*math.pi
                    v_possible = c*fprf/(4*math.pi*fc) * phi_possible
                    ts_possible=v_possible/((c/2)*fprf)
                    lag_possible=np.ceil(ts_possible*fs)
                    lag_possible=lag_possible.astype(int)
                    abs_lag_possible=abs(lag_possible)
                    np_size=len(np_range)
                    ccross = np.zeros((np_size,M-2),dtype=complex) # Pre-allocate matrix
                    cross_sum = np.zeros(np_size,dtype=complex)                    
                                        
                    for w in range(0,np_size): #search in np_range possible lags
                        for t in range(0,M-2):                        
                            if lag_possible[w]<0:                            
                                #ccross[w,t]  = np.dot(np.conj(dataDop[abs_lag_possible[w]:,t+1]),dataDop[:-abs_lag_possible[w],t])                                                
                                #ccross[w,t]  = np.vdot(dataDop[abs_lag_possible[w]:,t+1],dataDop[:-abs_lag_possible[w],t])
                                ccross[w,t]  = dataDop[abs_lag_possible[w]:,t+1].conj().T@dataDop[:-abs_lag_possible[w],t]
                            else:                    
                                if lag_possible[w]==0:                                
                                    #ccross[w,t]  = np.dot(np.conj(dataDop[:,t+1]),dataDop[:,t])                                
                                    #ccross[w,t]  = np.vdot(dataDop[:,t+1],dataDop[:,t])                                
                                    ccross[w,t]  = dataDop[:,t+1].conj().T@dataDop[:,t]                                

                                else:#>0             
                                    #ccross[w,t]  = np.dot(np.conj(dataDop[:-abs_lag_possible[w],t+1]),dataDop[abs_lag_possible[w]:,t])                   
                                    #ccross[w,t]  = np.vdot(dataDop[:-abs_lag_possible[w],t+1],dataDop[abs_lag_possible[w]:,t])
                                    ccross[w,t]  = dataDop[:-abs_lag_possible[w],t+1].conj().T@dataDop[abs_lag_possible[w]:,t]
                        cross_sum[w]=ccross[w,:].sum()                
                    #do not use max function without abs as it does not consider complex numbers
                    abs_cross_sum=abs(cross_sum)
                    lag=np.argmax(abs_cross_sum)
                    #lag = max(cross_sum, key=np.abs)                

                    phi_true=phi_auto+np_range[lag]*2*math.pi
                    
                    flow[j,i]    = -c*fprf/(4*np.pi*fc) * phi_true  

                ## ACM
                else:
                    flow[j,i] = -c*fprf/(4*np.pi*fc) * phi_auto
                
        elapsed = time.time() - tt                
                               
        return flow,elapsed

def test():
    return 0
    

    
    

if __name__ == '__main__':
    test()
    # execute only if run as the entry point into the program
    #test_doppler_mask()    
