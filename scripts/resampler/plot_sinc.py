#!env python3
import sys
import numpy
import pylab

size = 16

sinc = []
for line in sys.stdin:
    sinc.append(float(line))
sinc = numpy.array(sinc)

smin = numpy.min(sinc)
smax = numpy.max(sinc)

s2 = []
for x in sinc:
    s2.append(int(round((x - smin) / (smax - smin) * 2**size)))

s3 = []
for i, x in enumerate(s2):
    y = x / 2**size * (smax - smin) + smin
    s3.append(y - sinc[i])

f, ax = pylab.subplots(nrows=3, ncols=1)

p = ax[0]
p.plot(sinc, '-', label='sinc')
p.grid(True)

p = ax[1]
p.plot(s2, '-', label='sinc')
p.grid(True)

p = ax[2]
p.plot(s3, '-', label='sinc')
p.grid(True)
p.ticklabel_format(style='plain')

pylab.show()
