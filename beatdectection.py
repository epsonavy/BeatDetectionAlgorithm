# Beat Detection for wav file
# Author: Pei Lian Liu
# coding: utf-8

from scipy.io import wavfile
from numpy import *
from matplotlib.pyplot import *

# The wav file you want to detect the beat
sr, src = wavfile.read("/Users/EPSO/Documents/CS185C/HW3/BPM131.wav")

src.shape, src.dtype

# Downmix
src_mono = sum(src.astype(float)/src.shape[1], axis=1).astype(src.dtype)
# Normalize
abs_max = max(abs(src_mono.min().astype(float)), abs(src_mono.max().astype(float)))
src_mono_norm = src_mono.astype(float) / abs_max

# Plot the Downmix an Normalize of the sample
plot(src_mono_norm)
grid()
show()


# Attention, this snippet of code may need longer time to run
sample = src_mono_norm[44100:44100*5 + 44100] # 5 seconds
sample_fft = np.fft.fft(sample.astype(float)) # FFT of complex signal
N = len(sample_fft)  # N = length of the sample
ta = sample_fft.real # ta[k] = real part
tb = sample_fft.imag # tb[k] = imaginary part
BPM = arange (60, 180, 5) # Check from 60BPM to 180BPM per step of 5
Ti = 60 * 44100 / BPM # Get Tis according to BPMs
AmpMax = 32767 # This number is according to the PDF
E_BPMc = [] # List of Energy of the BPMs
for start in range(0, len(Ti)):
    l = [0] * N
    j = [0] * N
    for k in range(0, N):
        if(k % Ti[start] == 0):
            l[k] = j[k] = AmpMax;
        else:
            l[k] = j[k] = 0

    tl = np.fft.fft(l).real
    tj = np.fft.fft(j).imag

    for k in range(0, N):
        eng = abs(complex(ta[k],tb[k]) * complex(tl[k],tj[k]))
    eng_sum = sum(eng)
    E_BPMc.append(eng_sum)
BPM_max = max(E_BPMc)

maxIndex = E_BPMc.index(max(E_BPMc))

# Show the beat per minutes on the diagram
xlabel('BPM')
ylabel('Energy of BPM')
plot(BPM,E_BPMc)
plot(BPM[maxIndex],E_BPMc[maxIndex],'ro')
show()



