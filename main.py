from multiprocessing.sharedctypes import Value
from scipy.fftpack import rfft, rfftfreq
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import sys


def to_mono(signal):
    """
    Convert a stereo signal to mono.

    Args:
        signal (numpy.array): signal to convert (either mono or stereo)

    Returns:
        numpy.array: mono signal
    """
    channels = 1
    try:
        channels = signal.shape[1]
    except IndexError:
        pass

    if channels == 2:
        signal = signal.sum(axis=1) / 2

    return signal


def load_signal(file_name):
    """
    Load a wav file and return the signal and the sample rate

    Args:
        file_name (string): path of the wav file

    Returns:
        signal (numpy.array): the signal loaded into memory
        sample_rate (int): the sample rate of the signal
        N (int): the number of samples in the signal
        secs (float): the duration of the signal in seconds
        Ts (float): the duration of a sample in seconds
    """
    fs_rate, signal = wavfile.read(file_name)

    #print("Sample rate/frequency:", fs_rate)

    channels = 1
    try:
        channels = signal.shape[1]
    except IndexError:
        pass

    #print("Number of channels:", channels)

    N = signal.shape[0]
    #print("Number of samples:", N)

    secs = N / float(fs_rate)
    #print("Length in seconds:", secs)

    Ts = 1.0/fs_rate  # sampling interval in time
    #print("Timestep between samples Ts", Ts)

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
            #print("clipping at", i)
            signal = signal[:i]
            break
    N = signal.shape[0]
    secs = N / float(fs_rate)
    return signal, N, secs


def signal_beginning(signal, fs_rate, secs):
    signal = np.array(signal, dtype=float)

    N = int(secs * fs_rate)

    return signal[:N], N


def signal_end(signal, fs_rate, secs):
    signal = np.array(signal, dtype=float)

    N = int(secs * fs_rate)

    return signal[signal.size-N:], N


def plot_signal(signal, N, secs, Ts):
    # Plot original
    # Plot processed
    # Plot frequency domain of processed
    # time vector as scipy arange field / numpy.ndarray
    t = np.arange(0, secs, Ts)
    FFT = abs(rfft(signal))
    FFT_side = FFT[range(int(N/2))]  # one side FFT range
    freqs = rfftfreq(signal.size, t[1]-t[0])
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


def find_strongest_freqs(amplitudes, frequencies, nb_top_freqs):
    amplitude_freqs = {}

    # for the amplitude-frequency graph
    for i in range(len(frequencies)):
        if amplitudes[i] not in amplitude_freqs:
            amplitude_freqs[amplitudes[i]] = list()
        amplitude_freqs[amplitudes[i]].append(frequencies[i])

    x = 0
    max_freqs = []
    while(x < nb_top_freqs):
        max_amp = max(amplitude_freqs)
        # compare all the frequncies with the current maximum amplitude to the ones already found
        deleteItems = []
        for currFrequencies in amplitude_freqs[max_amp]:
            for i in range(len(max_freqs)):
                for knownFrequencies in max_freqs[i]:
                    # if the difference between them is less than  10 remove this current frequency
                    if(currFrequencies > knownFrequencies-10) and currFrequencies < (knownFrequencies+10):
                        # amplitude_freqs[max_amp].remove(currFrequencies)
                        deleteItems.append(currFrequencies)
        for i in range(len(deleteItems)):
            # if freq is not in delete items, we keep it
            amplitude_freqs[max_amp] = [
                freq for freq in amplitude_freqs[max_amp] if freq not in deleteItems]
        # if there are still valid current frequncies, print them
        if amplitude_freqs[max_amp]:
            max_freqs.append(amplitude_freqs[max_amp])
            x = x+1
            print("Frequency ", x, ":")
            print("Max amplitude: ", max_amp)
            print("Frequency value:", amplitude_freqs[max_amp], "\n")
        del amplitude_freqs[max_amp]
    return max_freqs

    '''
    for i in range(nb_top_freqs):
    max_amp = max(amplitude_freq)
    print("Max amplitude", i, ":", max_amp)
    print("Frequencies", amplitude_freq[max_amp])
    del amplitude_freq[max_amp]
    '''


def frequency_domain(signal, sample_rate):
    fft_spectrum = rfft(signal)
    freq = rfftfreq(signal.size, d=1./sample_rate)
    fft_spectrum_abs = np.abs(fft_spectrum)

    return fft_spectrum_abs, freq


def get_avg_rate(signal, frequency, fs_rate, secs):
    # get the amplitude in the first few seconds
    initial_sig = signal_beginning(signal, fs_rate, 0.5)
    initial_sig_ft = frequency_domain(
        initial_sig, len(initial_sig), 0.5, 1/fs_rate)
    print("Initial signal:", initial_sig_ft)
    ini_amp = initial_sig_ft[frequency]
    # get amplitude in the last fft
    final_sig_fft = frequency_domain(signal, len(signal), secs, 1/fs_rate)
    final_amp = final_sig_fft[frequency]
    # get the avg rate
    avg_rate = (final_amp - ini_amp) / secs
    return avg_rate


def experiment():
    raw_signal, fs_rate, M, raw_len, Ts = load_signal(sys.argv[1])
    processed_signal, N, pro_len = process_signal(raw_signal, fs_rate)
    start_len = 0.5
    start_signal, O = signal_beginning(raw_signal, fs_rate, start_len)
    # last_half_sec()
    # first_half_sec()

    top_freq = find_strongest_freqs(processed_signal, N, pro_len, Ts, 5)

    # plot_signal(start_signal, O, start_len, Ts)

    start_ft = frequency_domain(start_signal, O, start_len, Ts)
    end_ft = frequency_domain(processed_signal, N, pro_len, Ts)
    # print("Frequency:", ft)
    print("Amplitude of top freq:", start_ft[round(top_freq[0][0], 1)])
    print("Amplitude of top freq:", end_ft[top_freq[0][0]])

    print("Average rate:", (end_ft[top_freq[0][0]] -
          start_ft[round(top_freq[0][0], 1)])/pro_len)

    # for i in range(len(top_freq)):
    #     print("Frequency:", top_freq[i])
    #     print("Avg rate:", get_avg_rate(
    #         processed_signal, top_freq[i][0], fs_rate, pro_len))


def to_dict(freq, amp):
    freq_dict = {}
    for i in range(len(freq)):
        freq_dict[freq[i]] = amp[i]
    return freq_dict


# def plot_signal():
#     plt.subplot(211)

#     t = np.arange(0, secs, Ts)

def plot_sig(signal, fs_rate):
    signal = to_mono(signal)

    plt.subplot(211)

    secs = signal.shape[0] / float(fs_rate)
    Ts = 1.0/fs_rate  # sampling interval in time

    t = np.arange(0, secs, Ts)

    plt.plot(t, signal)

    amp, freq = frequency_domain(signal, fs_rate)
    plt.subplot(212)
    plt.plot(freq, amp)

    plt.show()


def main():
    snippet_len = 0.5

    print("File name: ", sys.argv[1], "\n")
    print("Frequencies with maximum amplitude:", "\n")

    raw_signal, fs_rate, N, raw_len, Ts = load_signal(sys.argv[1])

    # Cuts the signal when it clips to prevent artifacts from distorting the
    # fourier transform
    processed_signal, M, pro_len = process_signal(raw_signal, fs_rate)

    # if O and P are the same - then rfftfreq(signal.size,
    # d=1./sample_rate) outputs the same values as so we can use the dictionary
    signal_beg, O = signal_beginning(processed_signal, fs_rate, snippet_len)

    signal_en, P = signal_end(processed_signal, fs_rate, snippet_len)
    plot_sig(signal_en, fs_rate)

    beg_sig_amps, beg_sig_freqs = frequency_domain(signal_beg, fs_rate)
    end_sig_amps, end_sig_freqs = frequency_domain(signal_en, fs_rate)

    beg = to_dict(beg_sig_freqs, beg_sig_amps)
    full = to_dict(end_sig_freqs, end_sig_amps)

    # Get the list of 5 most prominent frequencies after the orginal signal has
    # resoin the room
    top_freqs = find_strongest_freqs(end_sig_amps, end_sig_freqs, 5)

    print("Average convergence rate for max amplitude frequencies:")
    for i in range(len(top_freqs)):
        avg_convergence_rate = (
            full[top_freqs[i][0]] - beg[top_freqs[i][0]])/pro_len
        print("Frequency ", i+1, " convergence rate: ", avg_convergence_rate)

    # find_top_freqs(processed_signal, M, pro_len, Ts, 5)
    # plot_signal(raw_signal, N, raw_len, Ts)
    # plot_signal(processed_signal, M, pro_len, Ts)
    # experiment()
    print("\n\n\n")


if __name__ == "__main__":
    main()
