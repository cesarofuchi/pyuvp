"""
Ultrasound class for signal processing

"""

# Authors: Cesar Ofuchi <cesarofuchi@gmail.com>
#          Fabio Rizental Coutinho <fabiocoutinho@professores.utfpr.edu.br >
# License: MIT

import numpy as np
from scipy.signal import lfilter, hilbert
import scipy.signal as sci
import matplotlib as plt
import math
import glob
import html

class Ultrasound:
    
    

    def __init__(self,fs,fc,fprf,samples,nwaves,data):
        '''
        basic ultrasonic data
        Attributes
        ----------
        fs: integer number
            sampling rate
        fc: integer number
            transducer central frequeny
        fprf: integer number
            pulse repetition frequency
        samples: integer number
            number of samples
        nwaves: integer number
            number waves acquired
        data: matrix number
            data matrix with raw data (not IQ for now)
        '''
        self.fs=fs
        self.fprf=fprf
        self.fc=fc
        self.nwaves=nwaves
        self.samples=samples
        self.nwaves=nwaves        
        self.data=data
        
    

def test():
    return 0



if __name__ == '__main__':
    # execute only if run as the entry point into the program
    #test_tof() 
    #test_folder()
    test()