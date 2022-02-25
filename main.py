from scipy import fftpack
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


def to_mono(signal):
    l_audio = len(signal.shape)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    return signal


def load_signal(file_name):
    fs_rate, signal = wavfile.read(file_name)
    print("Sample rate/frequency:", fs_rate)
    l_audio = len(signal.shape)
    print("Number of channels:", l_audio)
    N = signal.shape[0]
    print("Number of samples:", N)
    secs = N / float(fs_rate)
    print("Length in seconds:", secs)
    Ts = 1.0/fs_rate  # sampling interval in time
    print("Timestep between samples Ts", Ts)
    return signal, fs_rate, N, secs, Ts

# Cleans the signal by making sure it is a mono signal and removing any part of
# the signal beyond the first sign of clipping


def process_signal(signal, fs_rate):
    signal = np.array(signal, dtype=float)
    signal = to_mono(signal)
    nmax = max(signal)
    nmin = min(signal)
    for i in range(len(signal)):
        if signal[i] == nmax or signal[i] == nmin:
            print("clipping at", i)
            signal = signal[:i]
            break
    N = signal.shape[0]
    secs = N / float(fs_rate)
    return signal, N, secs


def plot_signal(signal, N, secs, Ts):
    # Plot original
    # Plot processed
    # Plot frequency domain of processed
    # time vector as scipy arange field / numpy.ndarray
    t = np.arange(0, secs, Ts)
    FFT = abs(fftpack.fft(signal))
    FFT_side = FFT[range(int(N/2))]  # one side FFT range
    freqs = fftpack.fftfreq(signal.size, t[1]-t[0])
    freqs_side = freqs[range(int(N/2))]  # one side frequency range
    plt.subplot(211)
    p1 = plt.plot(t, signal, "g")  # plotting the signal
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(212)
    # plotting the positive fft spectrum
    p3 = plt.plot(freqs_side, abs(FFT_side), "b")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count single-sided')
    plt.show()


def find_top_freqs(signal, N, secs, Ts, nb_top_freqs):
    t = np.arange(0, secs, Ts)
    FFT = abs(fftpack.fft(signal))
    FFT_side = FFT[range(int(N/2))]  # one side FFT range
    freqs = fftpack.fftfreq(signal.size, t[1]-t[0])
    freqs_side = freqs[range(int(N/2))]  # one side frequency range

    amplitude_freq = {}

    for i in range(len(FFT_side)):
        if FFT_side[i] not in amplitude_freq:
            amplitude_freq[FFT_side[i]] = list()
        amplitude_freq[FFT_side[i]].append(freqs_side[i])

    for i in range(nb_top_freqs):
        max_amp = max(amplitude_freq)
        print("Max amplitude:", i, ":", max_amp)
        print("Frequencies:", amplitude_freq[max_amp])
        del amplitude_freq[max_amp]


def main():
    raw_signal, fs_rate, N, raw_len, Ts = load_signal(
        "singleSnare2s.wav")
    processed_signal, M, pro_len = process_signal(raw_signal, fs_rate)
    find_top_freqs(processed_signal, M, pro_len, Ts, 5)
    # plot_signal(raw_signal, N, raw_len, Ts)
    plot_signal(processed_signal, M, pro_len, Ts)


if __name__ == "__main__":
    main()
