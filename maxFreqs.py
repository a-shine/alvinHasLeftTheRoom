import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

samplingFrequency, signalData = wavfile.read('singleSnare2s.wav')


plt.title('Spectrogram')
Pxx, freqs, bins, im = plt.specgram(
    signalData[1], Fs=samplingFrequency, NFFT=50000)
plt.xlabel('Time')
plt.ylabel('Frequency')

frequency_bin = freqs[1]-freqs[0]
a = []
for i in range(25001):
    a.append(Pxx[i][len(Pxx[0])-1])


N = 7

max_amp_end = np.argsort(a)[-N:]

ind = np.argpartition(a, -N)[-N:]
max_freq_end = []
for i in range(len(ind)):
    max_freq_end.append(ind[i]*frequency_bin)
print('max freq end')
print(max_freq_end)

b = []
for i in range(25001):
    b.append(Pxx[i][0])

ind2 = np.argpartition(b, -N)[-N:]
max_freq_start = []
for i in range(len(ind2)):
    max_freq_start.append(ind2[i]*frequency_bin)

print('max freq start')
print(max_freq_start)

max_bin_end = max_freq_end/frequency_bin

max_amp_end = []
for i in range(N):
    max_amp_end.append(Pxx[ind[i]][len(Pxx[0])-1])

print('max amp end')
print(max_amp_end)


max_amp_end_start = []
for i in range(N):
    max_amp_end_start.append(Pxx[ind[i]][0])
print('max amp from end at start')
print(max_amp_end_start)
conv = []
for i in range(N):
    conv.append((max_amp_end[i] - max_amp_end_start[i])/(len(Pxx[0])))
print('convergence rate')
print(conv)
