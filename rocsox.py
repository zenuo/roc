#!env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy.signal import *
from scipy.fftpack import *
from optparse import OptionParser
import wavio
import scipy.io.wavfile
import scipy
import math
import sys


def Spectrum(s):
    Ftest = scipy.fftpack.fft( s )
    n = round(s.shape[0]/2)
    xf = np.linspace(0.0, Fs/2.0, n)
    return xf, 20*np.log10(np.abs(Ftest[0:n])/n)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option( "-r", "--roc", action="store", 
                        default="/tmp/delta_out.wav",
                        type="string", dest="roc_file",
                        help="Where to store retrieved signal.")

    parser.add_option( "-s", "--sox", action="store", 
                        default="/tmp/delta_sox.wav",
                        type="string", dest="sox_file",
                        help="Where to store retrieved signal.")
    
    (options, args) = parser.parse_args()

    print("Openning {}".format(options.roc_file))
    print("Openning {}".format(options.sox_file))
    
    # out_fl = wavio.read( options.src_file )
    # out = np.array(out_fl.data[0:,0]/math.pow(2.0, out_fl.sampwidth*8-1))
    # out = out - np.mean(out)
    # Fs = Fso

    roc_fl = scipy.io.wavfile.read( options.roc_file )
    Fs_roc = roc_fl[0]
    roc = np.array([x/math.pow(2.0,31) for x,y in roc_fl[1]])

    print(roc_fl)
    
    sox_fl = scipy.io.wavfile.read( options.sox_file )
    Fs_sox = sox_fl[0]
    sox = np.array([x/math.pow(2.0,31) for x,y in sox_fl[1]])
    # out = np.array([x for x,y in out_fl[1]])
    print(sox_fl)

    roc_max = np.argmax(roc)
    sox_max = np.argmax(sox)
    print("roc_max: {}\nsox_max: {}".format(roc_max, sox_max))

    print("Fs: ", Fs_roc)
    Fs = Fs_roc

    width = 10000

    plt.figure()
    plt.subplot(211)
    # plt.plot(20*np.log10(sox[sox_max-width:sox_max+width]), label="sox")
    # plt.plot(20*np.log10(roc[roc_max-width:roc_max+width]), label="roc")
    plt.plot(sox[sox_max-width:sox_max+width], label="sox")
    plt.plot(roc[roc_max-width:roc_max+width], label="roc")
    plt.legend()
    plt.grid()
    # plt.xlim([250, 400])
    # plt.ylim([-0.025, 0.025])
    plt.subplot(212)
    plt.plot(*Spectrum(sox))
    plt.plot(*Spectrum(roc))
    plt.ylim([-150, 0])
    plt.show()
