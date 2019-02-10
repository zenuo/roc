#!env python3
import numpy
import pylab
import scipy.io.wavfile

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

p1.plot(original, '-', label='original')
p1.plot(resampler_cpy, '.', label='resampler dummy (copy)')
p1.plot(resampler_sc1, '-', label='resampler with scaling=1')
p1.legend(loc='lower left')

p2.plot(original - resampler_cpy, label='original - resampler dummy')
p2.legend(loc='lower left')

p3.plot(original - resampler_sc1, label='original - resampler scaling=1')
p3.legend(loc='lower left')

pylab.show()
