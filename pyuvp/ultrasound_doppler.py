""" ultrasound doppler algorithms

"""


# Authors: Cesar Ofuchi <cesarofuchi@gmail.com>
#          Fabio Rizental Coutinho <fabiocoutinho@professores.utfpr.edu.br >
# License: MIT

import pyuvp.ultrasound as usp
from scipy.signal import lfilter, hilbert,medfilt2d,correlate
import numpy as np
import math
import time

class UltrasoundDoppler:
    

    def __init__(self,usdata,ns,nc,ovs,ovt,cycles=4):
        '''
        Attributes
        ----------
        ns: integer number
            spatial resolution
        nc: integer number
            temporal resolution
        ovs: integer number
            spatial overlap 
        ovt: integer number
            temporal overlap    
        cycles: integer number
            number of cycles of transducer 
        '''
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
        # Hilbert axis=0 is utterly important!
        analytic_h=hilbert(filtered_data,axis=0)
        tp=np.arange(0,len(analytic_h),1)
        samples,pulses=filtered_data.shape
        
        #create time vector init=0 : end= (samples -1 )/fs : step= 1/fs 
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
        '''
        Autocorrelation reference:
        Kasai, C.; Namekawa, K.; Koyano, A.; Omoto, R. 
        Real-Time Two-Dimensional Blood Flow Imaging Using an Autocorrelation Technique. 
        IEEE Trans. Son. Ultrason. 1985, 32, 458–464.

        Loupas, T.; Powers, J.T.; Gill, R.W. 
        Axial velocity estimator for ultrasound blood flow imaging, based on a full evaluation of the Doppler equation by means of a two-dimensional autocorrelation approach. 
        IEEE Trans. Ultrason. Ferroelectr. Freq. Control 1995, 42, 672–688.

        Extended Autocorrelation reference:
        Ofuchi, C.Y.; Coutinho, F.R.; Neves, F.; De Arruda, L.V.R.; Morales, R.E.M. 
        Evaluation of an Extended Autocorrelation Phase Estimator for Ultrasonic Velocity Profiles Using Nondestructive Testing Systems.
        Sensors 2016, 16, 1250. https://doi.org/10.3390/s16081250

        Attributes
        ----------
        c: number
            sound speed in meters/second in the media

        np_range: integer number
            optional parameter used for method='extended' for the extended autocorrelation method. It is a search value range for velocities over Nyquist limit.
            Default value is from -2 to 2 times the Nyquist limit 

        method: string    
            optional parameter to select 'normal' or 'extedend' autocorrelation method. Default value is normal.
        '''
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
            self.iq= np.array([self.iq,pad])	
        
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

def staggered_trigger(self,c,m,n,T1,T2,debug=False):
    '''
    staggered trigger function based on Torres (2004)
    Torres S M & Dubel Y: Design, implementation, and demonstration of a staggered PRT algorithm for the WSR-88D. 
    J. Atmos. Oceanic Tech. 21 (2004), 1389-1399. 

    Attributes
    ----------
    c: number
        sound speed in meters/second in the media

    m, n, T1 and T2: used to have a ratio of the 2 different fprf. It is 2/3, 4/5 , 5/6 ...
        
    
    
    '''
    
    nc=self.nc
    ns=self.ns        
    ovs=self.ovs        
    ovt=self.ovt        
    fprf=self.usdata.fprf
    fc=self.usdata.fc
    fs=self.usdata.fs

    
    #n=3
    #m=2    
    #fprf=2000
    #T2=n/fprf
    #T1=m/fprf
    #TODO add noise analysis
    #noisepw=0.59
    #ns=50;nc=50;ovs=1;ovt=1;
    #np_range=np.array(np_range)
            
    samples,pulses = self.iq.shape

    rest=np.remainder(samples*-ns,ns*ovs)
    if rest !=0:
        pad_n=int(ns*ovs-rest)
        pad=np.zeros(pad_n,pulses)
        self.iq= np.array([self.iq,pad])	

    nchannels=samples/(ovs*ns)
    nchannels=int(nchannels)
    nchannelst=np.floor((pulses-nc)/(ovt*nc))+1
    nchannelst=int(nchannelst)

    # staggered
    va1 = c/(4*T1*fc)#  vmax T1
    va2 = c/(4*T2*fc)#  vmax T2

    C0=0
    C1=C0+2*va2
    C2=C1-2*va1
    C3=C2+2*va2
    C4=C3-2*va1
    C5=C4+2*va2
    C6=C5-2*va1

    ##
    #init results zeros ()
    # need to clean the useful results from debug results
    v1d=np.zeros((nchannels,nchannelst)) # dealised velocity1
    v2d=np.zeros((nchannels,nchannelst)) # dealised velocity2

    v12=np.zeros((nchannels,nchannelst)) # velocity difference (aliased) radial
    
    #TODO add noise analysis
    #snr1=np.zeros((nchannels,nchannelst)) # snr1
    #snr2=np.zeros((nchannels,nchannelst)) # snr2
    #sigmaT1=np.zeros((nchannels,nchannelst)) # velocity variance from autocorrelation
    #sigmaT2=np.zeros((nchannels,nchannelst)) # velocity variance from autocorrelation
    #W=np.zeros((nchannels,nchannelst)) # power    

    v1=np.zeros((nchannels,nchannelst)) # 
    v2=np.zeros((nchannels,nchannelst)) # 
    vst=np.zeros((nchannels,nchannelst)) # 

    vr=np.zeros((nchannels,nchannelst))
    desv=np.zeros((nchannels,nchannelst))

    tt = time.time()
    # iterate through all temporal channels
    for i in range(nchannelst):    
        p_i=i*nc*ovt
        p_f=p_i+nc
        # iterate through all spatial channels
        for j in range(nchannels):       
        #for j in range(1):       
            ps_i=(j)*ns*ovs
            ps_f=ps_i+ns
            dataDop=self.iq[ps_i:ps_f,p_i:p_f]
            Nsamples,M=dataDop.shape                
            #sum all data (is equivalent to process all lines)
            vdata=np.sum(dataDop,axis=0)           

            
            Nc1= vdata[1:-1:2].shape[0]
            Nc2= vdata[2:-1:2].shape[0]
            

            # accordign to Loupas1995 pag. 5 eq. 21 and further
            # Calculate the autocorrelation of R(T1) and R(T2)
            auto1T1= (vdata[1:-1:2].dot(vdata[0:-2:2].conj()))/Nc1  # R(T1)
            auto1T2= (vdata[2:-1:2].dot(vdata[1:-2:2].conj()))/Nc2  # R(T1)        
            
            # Calculate the autocorrelation of RT1(0) and RT2(0)            
            auto0T1=(np.sum(abs(vdata[1:-1:2])**2)+np.sum(abs(vdata[0:-1:2])**2)) /(2*Nc1)   # R(0) = W
            auto0T2=(np.sum(abs(vdata[2:-1:2])**2)+np.sum(abs(vdata[1:-1:2])**2)) /(2*Nc2)   # R(0) = W

            #W[j,i] = (auto0T1+auto0T2)/2
            
            
            
            #TODO add noise analysis
            #gamaT1=(abs(auto1T1)**2)/(abs(auto0T1)**2)
            #gamaT2=(abs(auto1T2)**2)/(abs(auto0T2)**2)

            #SNRT1=abs(auto0T1)/abs(noisepw)
            #SNRT2=abs(auto0T2)/abs(noisepw)

        
            #  Calculate the standard deviation of the estimate
            #sigmaT1[j,i]= (c/(2*fc))*np.sqrt(((1+1/SNRT1)**2-gamaT1)/(8*Nc1*math.pi**2*T1**2*gamaT1))
            #sigmaT2[j,i]= (c/(2*fc))*np.sqrt(((1+1/SNRT2)**2-gamaT2)/(8*Nc2*math.pi**2*T2**2*gamaT2))                              

            #  Calculate the phase angle from the autocorrelation of lag 1
            phi_est1 =  np.angle(auto1T1)   
            phi_est2 =  np.angle(auto1T2)           
            
            #snr1[j,i]=10*math.log10(SNRT1)
            #snr2[j,i]=10*math.log10(SNRT2)
            
            # calculate the axial velocity
            v1[j,i] = -c*(1/(T1))/(4*math.pi*fc) *  phi_est1
            v2[j,i] = -c*(1/(T2))/(4*math.pi*fc) *  phi_est2       
            v12[j,i]= v1[j,i]-v2[j,i]; 
            vst[j,i]=-c*(1/abs(T1-T2))/(4*math.pi*fc) * (phi_est2-phi_est1)

            ######################
            
    # end of loop
    # apply dealising rules acoording Torres et al (2004)
    va=m*va1
    dC=2*va/(m*n)

    # m=0
    rule=(v12>(C0-dC/2))&(v12<(C0+dC/2))
    P=0;Q=0;
    v1d[rule]=v1[rule]+2*P*va1
    v2d[rule]=v2[rule]+2*Q*va2


    if m>=1:
        rule=v12>(C1-dC/2)
        P=0;Q=1;
        v1d[rule]=v1[rule]+2*P*va1
        v2d[rule]=v2[rule]+2*Q*va2
        #
        rule=v12<-(C1-dC/2)
        P=0;Q=-1;
        v1d[rule]=v1[rule]+2*P*va1
        v2d[rule]=v2[rule]+2*Q*va2

    if m>=2:
        rule=(v12>(C2-dC/2))&(v12<(C2+dC/2))
        P=1;Q=1;
        v1d[rule]=v1[rule]+2*P*va1
        v2d[rule]=v2[rule]+2*Q*va2
        #
        rule=(v12<-(C2-dC/2))&(v12>-(C2+dC/2))
        P=-1;Q=-1;
        v1d[rule]=v1[rule]+2*P*va1
        v2d[rule]=v2[rule]+2*Q*va2


    if m>=3:
        rule=(v12>(C3-dC/2))&(v12<(C3+dC/2))
        P=1;Q=2;
        v1d[rule]=v1[rule]+2*P*va1
        v2d[rule]=v2[rule]+2*Q*va2
        #
        rule=(v12<-(C3-dC/2))&(v12>-(C3+dC/2))
        P=-1;Q=-2;
        v1d[rule]=v1[rule]+2*P*va1
        v2d[rule]=v2[rule]+2*Q*va2


    if m>=4:
        rule=(v12>(C4-dC/2))&(v12<(C4+dC/2))
        P=2;Q=2;
        v1d[rule]=v1[rule]+2*P*va1
        v2d[rule]=v2[rule]+2*Q*va2
        ##
        rule=(v12<-(C4-dC/2))&(v12>-(C4+dC/2))
        P=-2;Q=-2;
        v1d[rule]=v1[rule]+2*P*va1
        v2d[rule]=v2[rule]+2*Q*va2


    if m>=5:
        rule=(v12>(C5-dC/2))&(v12<(C5+dC/2))
        P=2;Q=3;
        v1d[rule]=v1[rule]+2*P*va1
        v2d[rule]=v2[rule]+2*Q*va2
        ##
        rule=(v12<-(C5-dC/2))&(v12>-(C5+dC/2))
        P=-2;Q=-3;
        v1d[rule]=v1[rule]+2*P*va1
        v2d[rule]=v2[rule]+2*Q*va2


    if m>=6:
        rule=(v12>(C6-dC/2))&(v12<(C6+dC/2))
        P=3;Q=3;
        v1d[rule]=v1[rule]+2*P*va1
        v2d[rule]=v2[rule]+2*Q*va2
        #
        rule=(v12<-(C6-dC/2))&(v12>-(C6+dC/2))
        P=-3;Q=-3;
        v1d[rule]=v1[rule]+2*P*va1
        v2d[rule]=v2[rule]+2*Q*va2

    elapsed = time.time() - tt   
    vr=(v1d+v2d)/2

    return vr

def test():
    return 0
    

    
    

if __name__ == '__main__':
    test()
    # execute only if run as the entry point into the program
    #test_doppler_mask()    
