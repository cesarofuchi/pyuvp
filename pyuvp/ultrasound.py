import numpy as np
from scipy.signal import lfilter, hilbert
import scipy.signal as sci
import matplotlib as plt
import math
import glob
import html
"""
Ultrasound class for signal processing

"""
class Ultrasound:
    
    

    def __init__(self,fs,fc,fprf,samples,nwaves,data):
        '''
        basic ultrasonic data
        '''
        self.fs=fs
        self.fprf=fprf
        self.fc=fc
        self.nwaves=nwaves
        self.samples=samples
        self.nwaves=nwaves        
        self.data=data
        
    def plotA(self,range1=0,wave=0):
        sx,sy=self.data.shape
        fig, ax=plt.subplots(figsize=(13, 9))
        ax.set_title('A scan')
        ax.set_ylabel('pulses')
        ax.set_xlabel('samples')
        plt.plot(self.data[range1,wave])
        

    def plotB(self,range1=0,range2=0):
        # define colormap
        #light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)
        sx,sy=self.data.shape
        #if range1[0]==0:
        if type(range1)==int:
            range1=np.arange(0,sx)
        if type(range2)==int:
            range2=np.arange(0,sy)
        if range1[-1]>sx or range2[-1]>sy:
             raise ValueError("range is beyond data size")
        cm = plt.cm.get_cmap("jet")
        fig, ax=plt.subplots(figsize=(13, 9))
        ax.set_title('B mode')
        ax.set_ylabel('pulses')
        ax.set_xlabel('samples')

        sc=ax.imshow(self.data[range1[0]:range1[-1],range2[0]:range2[-1]],cmap=cm,aspect='auto')
        fig.colorbar(sc, ax=ax, orientation="horizontal")
        plt.show()   

def tof(data,range1,range2,thr,method='max_peak',debug=False):    #try vargin after       

    if len(data)<range2[-1]:
         raise ValueError("range2 is beyond data size" % range2)

    #data from the reference echo
    data1=abs(hilbert(data[range1]))
    max1 = np.amax(data1, axis=0)
    pos1 = np.argmax(data1, axis=0)
    pos1=pos1+range1[0]

    data2=abs(hilbert(data[range2]))

    #TODO add selection of first peak
    if method in ['max_peak']:
        max2 = np.amax(data2, axis=0)        
        pos2 = np.argmax(data2, axis=0)
        pos2=pos2+range2[0]

    elif method in ['first_peak']:
        peaks,properties=sci.find_peaks(data2,prominence=thr,width=20)
        if not peaks: # didn't find ...
            pos2=-1
            max2=0
        else:
            pos2=peaks[0]+range2[0]
            max2=data2[peaks[0]]
            if debug:
                fig = plt.figure(figsize=(15, 10))
                plt.plot(data2)
                plt.plot(peaks, data2[peaks], "x")
                ax1.vlines(x=peaks, ymin=data2[peaks] - properties["prominences"],
                ymax = data2[peaks], color = "C1")
                ax1.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
                xmax=properties["right_ips"], color = "C1")  
    else:
        raise ValueError("method must be 'first_peak' or 'max_peak'."
                         % method)
    
    if max2<thr:
        pos2=-1
    
    return pos1,max1,pos2,max2,pos2-pos1
    
    
def get_tof_time_series(data,range1,range2,thr):

    list = []
    samples,ndata=data.shape   
    for i in range(1,ndata):
        ret=tof(data[:,i],range1,range2,thr)
        if(ret[2]>0):
            list.append(ret)
        else:
            n_ret=(ret[0],ret[1],range2[-1],ret[3],range2[-1]-ret[0])
            list.append(n_ret)
    
    
    np_list=np.array(list)
    return np_list

def get_tof_distance(samples,fs,c,unit='mm'):
    """
    Returns the distance of a echo-pulse tof
        ``c*(samples/fs)/2.``
    
    Parameters
    ----------
    samples : data in samples of the distance
        Input array.
    fs : ADC sampling frequency (e.g. Hz) 50 MHz        
    c : speed of sound in the medium in m/s
        (ex. c=1500 m/s in water)    
    unit : string, optional
        can be {'meter' or 'mm'}.  Default is mm.
        
    Returns
    -------
    distance : ndarray
        numpy array of distances    
    """
    
    if unit in ['mm']:
        unit_conv=1000
    elif unit in ['meter']:
        unit_conv=1
    else:
        raise ValueError("unit must be 'meter' or 'mm'."
                         % unit)
    distance=unit_conv*c*(samples/fs)/2
    return distance



def energy_diff(data,time_window,spatial_window,time_ini=0,spatial_ini=0):
    time_length,spatial_length=data.shape
    
    filter_data=np.zeros((int(np.round(time_length/time_window)),int(np.round(spatial_length/spatial_window))))
    tIter=0
    for t in range(time_ini,time_length-time_window,time_window):
        sIter=0
        for s in range(spatial_ini,spatial_length-spatial_window,spatial_window):
            dif=abs(data[t+time_window,s:s+spatial_window])-abs(data[t,s:s+spatial_window])
            filter_data[tIter,sIter]=dif.mean()
            sIter=sIter+1
        tIter=tIter+1

    
    strTitle="filtered: timeW="+str(time_window)+" spaceW="+str(spatial_window)

    cm = plt.cm.get_cmap("jet")
    fig, ax=plt.subplots(1,2,figsize=(13, 9))
    ax[0].set_title('original')
    ax[0].set_ylabel('pulses')
    ax[0].set_xlabel('samples')
    sc1=ax[0].imshow(abs(data),cmap=cm,aspect='auto')
    fig.colorbar(sc1, ax=ax[0], orientation="vertical")

    ax[1].set_title(strTitle)
    ax[1].set_ylabel('pulses')
    ax[1].set_xlabel('samples')
    sc2=ax[1].imshow(abs(filter_data),cmap=cm,aspect='auto',interpolation='none', extent=[1,spatial_length,1,time_length])


    fig.colorbar(sc2, ax=ax[1], orientation="vertical")
    plt.show()    

    return filter_data

def energy(data,ax=1):
    e=np.power(abs(hilbert(data)), 2)
    en=np.sum(e,axis=ax)
    return en

def test():
    return 0



if __name__ == '__main__':
    # execute only if run as the entry point into the program
    #test_tof() 
    #test_folder()
    test()