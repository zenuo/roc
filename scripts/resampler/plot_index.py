#!env python3
import sys
import numpy
import pylab

bins = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

raw_idx = []
for line in sys.stdin:
    raw_idx.append(int(line))
raw_idx = numpy.array(raw_idx)

raw_diff = numpy.hstack(([0], numpy.diff(raw_idx)))
raw_step_idx = numpy.where(raw_diff>0)
raw_step_x = numpy.arange(0,len(raw_idx),1)[raw_step_idx]
raw_step_y = raw_diff[raw_step_idx]

xticks = numpy.arange(0,len(raw_idx),10*64*2)

tbl_sstep = 40
tbl_mstep = 360/4
tbl_lstep = 360
tbl_cnt = 8

def tblshift(tn, idx):
    return idx - (tn*tbl_sstep)

def tblidx(tn, idx):
    pos = tblshift(tn, idx)
    return pos // tbl_lstep * tbl_mstep + pos % tbl_lstep

def intbl(tn, idx):
    if tn < 0:
        return False
    pos = tblshift(tn, idx)
    return pos % tbl_lstep < tbl_mstep

def findtbl(idx):
    mintn = None
    for tn in range(tbl_cnt):
        if intbl(tn, idx):
            if mintn is None or tblshift(tn, idx) < tblshift(mintn, idx):
                mintn = tn
    return mintn

tbl_num = []
tbl_idx = []

tbl_idx_last = [0]*tbl_cnt
tbl_idx_diff = [0]

tbl_num_cur = -1

for idx in raw_idx:
    if not intbl(tbl_num_cur, idx):
        tbl_num_cur = findtbl(idx)

    tbl_idx_cur = tblidx(tbl_num_cur, idx)

    tbl_num.append(tbl_num_cur)
    tbl_idx.append(tbl_idx_cur)

    tbl_idx_diff.append(tbl_idx_cur - tbl_idx_last[tbl_num_cur])
    tbl_idx_last[tbl_num_cur] = tbl_idx_cur

tbl_idx_diff = numpy.array(tbl_idx_diff)

f, ax = pylab.subplots(nrows=4, ncols=2)

p = ax[0, 0]
p.plot(raw_idx, '.', label='raw_idx')
p.set_xticks(xticks)
p.grid(True)
p.legend(loc='upper left')

p = ax[1, 0]
p.plot(raw_diff, '.-', label='raw_diff')
p.set_xticks(xticks)
p.grid(True)
p.legend(loc='upper left')

p = ax[2, 0]
p.plot(raw_step_x, raw_step_y, '.-', label='raw_step')
p.set_xticks(xticks)
p.set_yticks(numpy.arange(numpy.min(raw_step_y),numpy.max(raw_step_y),100))
p.grid(True)
p.legend(loc='upper left')

p = ax[3, 0]
p.hist(raw_diff, bins=bins, rwidth=0.9, label='raw_hist')
p.set_xscale('log', basex=2)
p.grid(True)
p.legend(loc='upper left')

p = ax[0, 1]
p.plot(tbl_idx, '.', label='tbl_idx')
p.set_xticks(xticks)
p.grid(True)
p.legend(loc='upper left')

p = ax[1, 1]
p.plot(tbl_idx_diff, '.-', label='tbl_diff')
p.set_xticks(xticks)
p.grid(True)
p.legend(loc='upper left')

p = ax[2, 1]
p.plot(tbl_num, '.', label='tbl #')
p.set_xticks(xticks)
p.set_yticks(numpy.arange(0,tbl_cnt,1))
p.grid(True)
p.legend(loc='upper left')

p = ax[3, 1]
p.hist(tbl_idx_diff, bins=bins, rwidth=0.9, label='tbl_hist')
p.set_xscale('log', basex=2)
p.grid(True)
p.legend(loc='upper left')

if len(sys.argv) != 1:
    pylab.gcf().set_size_inches(18.5, 10.5)
    pylab.savefig('%d.png' % tbl_stp, bbox_inches='tight')
else:
    pylab.show()
