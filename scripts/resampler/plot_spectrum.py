#!env python3
import numpy
import pylab
import scipy.fftpack
import scipy.io.wavfile

def spectrum(s):
    rate = 48000
    Ftest = scipy.fftpack.fft( s )
    n = round(s.shape[0]/2)
    xf = numpy.linspace(0.0, rate/2.0, n)
    return xf, 20*numpy.log10(numpy.abs(Ftest[0:n])/n)

original      = scipy.io.wavfile.read('mkh10/original.wav'     )[1][:,0]
resampler_cpy = scipy.io.wavfile.read('mkh10/resampler_cpy.wav')[1][:,0]
resampler_sc1 = scipy.io.wavfile.read('mkh10/resampler_sc1.wav')[1][:,0]

original      = original      / 2**15
resampler_cpy = resampler_cpy / 2**31
resampler_sc1 = resampler_sc1 / 2**31

# resampler cuts first and last 1024 samples
original = original[1024:]
original = original[:len(original)-1024]

# sox cuts some more at the end
original = original[:len(resampler_cpy)]

f, [p1, p2, p3] = pylab.subplots(nrows=3, ncols=1, sharex=True, sharey=True)

p1.plot(*spectrum(original), label='original')
p1.legend(loc='upper center')

p2.plot(*spectrum(original), label='original')
p2.plot(*spectrum(resampler_cpy), label='resampler dummy (copy)')
p2.legend(loc='upper center')

p3.plot(*spectrum(original), label='original')
p3.plot(*spectrum(resampler_sc1), label='resampler with scaling=1')
p3.legend(loc='upper center')

pylab.show()
