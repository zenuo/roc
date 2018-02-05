#!env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy.signal import *
from scipy.fftpack import *
from optparse import OptionParser
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
    parser.add_option( "-f", "--file", action="store", 
                        default="/tmp/sin.wav",
                        type="string", dest="dest_file",
                        help="Where to store retrieved signal.")

    (options, args) = parser.parse_args()

    Fs = 48000
    F = 440.0
    SNR = 140 # dB
    A = [8000, 1000]
    freq=[440.0, 23e3]
    signal_power = np.square(np.linalg.norm(A))
    noise_sigma = np.sqrt(signal_power / math.pow( 10, SNR/20 ))

    def fan(time):
        f = sum([a*np.sin(2*np.pi*f*time) for a,f in zip(A, freq)])
        f = f + np.random.normal(0, noise_sigma)
        return f

    t = np.arange( 5.0, step = 1/Fs )
    y = np.array([[fan(_),0] for _ in t], np.int16)
    # y = np.zeros([t.shape[0],2],np.int16)
    # y[round(44100*2.5), 0] = 16384

    scipy.io.wavfile.write(options.dest_file, Fs, y)

    y = y.T

    plt.figure()
    plt.subplot(211)
    plt.plot(*Spectrum(y[0]))
    plt.grid()
    plt.xlim([0, 24e3])
    plt.subplot(212)
    plt.plot(y[0])
    plt.show()
