#!/usr/bin/env python3

import sys
import math
import numpy as np
from numpy import argmax, sqrt, mean, absolute, arange
from numpy.fft import fft
from matplotlib.pyplot import *
#from scipy.signal import blackmanharris
from scipy.signal import *
#from scipy.signal.windows import hann

# window selection


def log10(x):
    x = np.where(x == 0, 1e-100, x)    
    return np.log10(x)

def get_window(window, ns):
    if window == "rect":
        w = np.ones(ns)
        main_lobe_width = 2
        print("Rectangular window")
    elif window == "hanning":
        w = np.hanning(ns)
        main_lobe_width = 4
    elif window == "blackman":
        w = np.blackman(ns)
        main_lobe_width = 6
    elif window == "blackmanharris":
        w = blackmanharris(ns)
        main_lobe_width = 10
    elif window == "hamming":
        w = np.hamming(ns)
        main_lobe_width = 4
    elif window == "flattop":
        w = flattop(ns)
        main_lobe_width = 12
    elif window == "kaiser":
        w = np.kaiser(ns, 30)
        main_lobe_width = 20
    else:
        raise ValueError("Unknown window type %s" % window)

    return w, main_lobe_width


def do_2s_complement(y, nb):
    for i in range(len(y)):
        if y[i] < 0:
            y[i] = 2**nb + y[i]
    return y


def undo_2s_complement(y, nb):
    for i in range(len(y)):
        if y[i] >= 2 ** (nb - 1):
            y[i] = -(2**nb) + y[i]
    return y


def spectrum(y):
    Y = np.fft.fft(y) / len(y)
    return Y


def tplot(y, fs, ptitle=""):
    N = range(len(y))
    t = np.array(N) * (1000 / fs)
    xlabel("Time [ms]")
    ylabel("Full-scale amplitude")
    title(ptitle)
    plot(t, y)
    show()


def fplot(y, fs, ptitle="", window="kaiser"):
    w, main_lobe_width = get_window(window, len(y))
    yw = y * w
    Yw = spectrum(yw)
    N = range(len(y))
    f = np.array(N) * (fs / len(y) / 1000)
    xlabel("Frequency [kHz]")
    ylabel("Full-scale amplitude")
    title(ptitle)
    plot(f, 20 * log10((abs(Yw))))
    show()


def pfplot(Yw, fs, ptitle=""):
    N = range(len(Yw))
    f = np.array(N) * ((fs / 2) / len(Yw) / 1000)
    xlabel("Frequency [kHz]")
    ylabel("Full-scale amplitude FF")
    title(ptitle)
    plot( 10 * log10((Yw)))
    show()


def thdn(y, fs, name, fmax=0, OSR=1, window="kaiser"):

    if fmax == 0:
        fmax = (fs/OSR) / 2
    print("fmax = ", fmax)
        
    small = 1e-100
    # Total signal+noise power before windowing
    Py = np.sum(y**2) / len(y)
    #print("Py: ", Py)

    # window
    w, main_lobe_width = get_window(window, len(y))
    yw = y*w

    Pyw = np.sum(y**2) / len(y)
    #print("Pyw = ", Pyw)

    # magnitude spectrum
    Y = spectrum(yw)

    # power spectrum with window correction
    PY = np.abs(Y) ** 2 * len(Y) / np.sum(w**2)
    
    # use half spectrum
    PY = 2 * PY[0 : len(Y) // 2]

    # compute power from power spectrum
    Py = np.sum(PY)
    print("Py = ", Py)
    #pfplot(PY, fs, name + " Power Spectrum")

    # signal power
    i = argmax(PY[0 : int(len(PY) * fmax / (fs/2))])
    r = np.arange(i - main_lobe_width//2, i + main_lobe_width//2)
    print("r(Ps) Ps ", r, PY[r])
    # tone frequency
    f = i * fs / len(y)
    print(fmax, fs, OSR, len(PY) * fmax / (fs/2))
    Ps = np.sum(PY[r])
    #pfplot(PY[r], fs, name + " Signal Spectrum")
    print("Ps = ", 10*log10(Ps))
    print("r, PY[r] = ", r, PY[r])

    #
    # noise and distortion  power
    #

    # exclude signal
    PY[r] = small

    # exclude 0-20Hz frequency range
    PY[range(0, int(20 * len(PY) / (fs / 2)))] = small
    #print(range(0, int(20 * len(PY) / (fs / 2))))

    # exclude frequencies above fmax
    PY[range(int(fmax * len(PY) / (fs / 2)), len(PY))] = small
    #print(range(int(fmax * len(PY) / (fs / 2)), len(PY)))
    #pfplot(PY, fs, name + " Noise Spectrum")

    # compute distortion and noise power
    Pn = np.sum(PY)
    #print("Pn, max PY, argmax", 10*log10(Pn), np.max(PY), np.argmax(PY))

    Ay = sqrt(2 * Ps)

    # signal to noise ratio
    thdn = -10 * log10(Ps / Pn)

    # equivalent number of bits
    enob = (thdn - 1.76) / 6.02

    print(
        "test=%s \t THD+N=%.1f dB \t f=%.4f Hz \t A=%.1f dB \t ENOB=%.1f bits"
        % (name, thdn, f, 20 * log10(Ay), enob)
    )


def limit_signal_freq(freqs, flimit):
    limits = np.full_like(freqs, flimit)
    return np.minimum(freqs, limits)


def tone(fs, f, AdB, ns, nc, fdelta, flimit):
    A = np.power(10, AdB / 20)
    t = np.array(range(ns), dtype="f8") * (1 / fs)
    pi = np.pi
    fa = f + np.array(range(nc)) * fdelta
    fa = limit_signal_freq(fa, flimit)
    x = A * np.cos(2 * pi * fa.reshape(nc, 1) * t)
    return x


def step(fs, f, AdB, ns):
    A = np.power(10, AdB / 20)
    t = np.array(range(ns), dtype="f8") * (1 / fs)
    x = A * np.heaviside(t, 1)
    return x


def impulse(fs, f, AdB, ns):
    A = np.power(10, AdB / 20)
    t = np.array(range(ns), dtype="f8") * (1 / fs)
    x = A * np.heaviside(t, 1)
    return x


def constant(fs, f, AdB, ns, nc, fdelta):
    """
    Generate constant audio signal for each channel.
    Linearly distributed amplitudes between channels.
    """
    A = np.power(10, AdB / 20)
    t = np.array(range(ns), dtype="f8") * (1 / fs)
    pi = np.pi
    fa = f + np.array(range(nc)) * fdelta
    x = A * np.cos(2 * pi * fa.reshape(nc, 1) * t)
    ch_As = np.arange(0, A, A / nc)
    for ch, A_ in zip(range(nc), ch_As):
        x[ch, :] = A_ * np.ones(ns)
    return x


def lpf(log2nz, log2size, nb):
    nz = 2**log2nz
    size = int(2**log2size)
    nb_f = log2size - log2nz
    t_step = np.double(1.0 / 2**nb_f)
    t = np.array(range(size), dtype="f8") * t_step
    w = (
        np.kaiser(2 * size - 1, 14.4) * 0.875
    )  # multiplies coeffs by anti-aliasing cutoff factor
    w = w[size - 1 : 2 * size - 1]  # take half window
    h = np.sinc(t) * w * 0.9999
    return h


def read_file(infile, fformat, nc, ns):
    fp = open(infile, "r")

    if fformat == "hex":
        lines = fp.readlines()
        y = np.arange(ns * nc)

        nb = 0
        for i in range(ns * nc):
            hex = lines[i]
            if nb == 0:
                nb = 4 * (len(hex) - 1)
            y[i] = int(hex, 16)
    else:  # TODO: debug this
        y = np.fromfile(infile, dtype="<i4")
        nb = int(np.ceil(np.log2(y.max())))

    y = np.double(undo_2s_complement(y, nb)) / np.double(2 ** (nb - 1))
    y = y.reshape(int(len(y) / nc), nc).T
    return y


def write_file(x, nb, outfile, fformat):
    if len(x.shape) > 1:
        nc, ns = x.shape
        x = x.T.reshape(
            nc * ns,
        )
    x = x * 2 ** (nb - 1)
    x = x.astype(int)
    x = do_2s_complement(x, nb)
    fp = open(outfile, "w")
    if fformat == "hex":
        np.savetxt(outfile, x.T, fmt="%06x", delimiter="\n")
    else:  # TODO: debug this
        # <i4: little-endian 4-byte integer
        x.astype("<i4").tofile(fp)


def print_usage():
    print("Usage: iob_audio 'parameter string'")
    print("       parameter string: 'param1=val1 param2=val2, ...'")
    print("       mode={tone|tplot|fplot|thdn|srct}")
    print("             tone: create sinusoidal audio signal")
    print("             lpf: generate low-pass FIR filter coeffs")
    print("             tplot: draw time plot")
    print("             fplot: draw frequency plot")
    print("             thdn: compute THD+N")
    print("             srct: create input test signal for sample rate conversion")
    print("       fs=<sample rate in Hz>")
    print("       nb=<sample number of bits>")
    print("       A=<amplitude in dBFS>")
    print("       ns=<number of samples>")
    print("       f=<signal frequency in Hz>")
    print("       fdelta=<frequency increment for next channel>")
    print("       flimit=<maximum limit for signal frequency>")
    print("       fin=<input sample rate in Hz>")
    print("       fout=<output sample rate in Hz>")
    print("       infile=<input file>")
    print("       outfile=<output file>")
    print("       fformat=<file format: hex or bin>")
    print("       test_name=<(optional) test_name>")
    print("Example:")
    print("> iob_audio 'mode=tone fs=44100 f=1000 ns=441'")
    exit(1)


def main():
    if len(sys.argv) != 2:
        print_usage()

    param_array = sys.argv[1].split()

    param_dict = {}
    for i in param_array:
        j = i.split("=")
        param_dict[j[0]] = j[1]

    # assign parameters
    if "mode" in param_dict:
        mode = param_dict["mode"]
    else:
        mode = "thdn"

    if "fs" in param_dict:
        fs = np.double(param_dict["fs"])
    else:
        fs = np.double(48000)

    if "ns" in param_dict:
        ns = int(param_dict["ns"])
    else:
        ns = 0

    if "log2nc" in param_dict:
        log2nc = int(param_dict["log2nc"])
        nc = 2**log2nc
    else:
        nc = 1

    if "fdelta" in param_dict:
        fdelta = np.double(param_dict["fdelta"])
    else:
        fdelta = np.double(1000)

    if "flimit" in param_dict:
        flimit = np.double(param_dict["flimit"])
    else:
        flimit = fs / 2  # nyquist frequency

    if "fin" in param_dict:
        fin = np.double(param_dict["fin"])
    else:
        fin = np.double(48000)

    if "fout" in param_dict:
        fout = np.double(param_dict["fout"])
    else:
        fout = np.double(48000)

    if "f" in param_dict:
        f = np.double(param_dict["f"])
    else:
        f = np.double(1000)

    if "AdB" in param_dict:
        AdB = np.double(param_dict["AdB"])
    else:
        AdB = np.double(-1)

    if "nb" in param_dict:
        nb = int(param_dict["nb"])
    else:
        nb = int(24)

    if "log2nz" in param_dict:
        log2nz = int(param_dict["log2nz"])
    else:
        log2nz = int(4)

    if "log2size" in param_dict:
        log2size = int(param_dict["log2size"])
    else:
        log2size = int(14)

    if "infile" in param_dict:
        infile = param_dict["infile"]
    else:
        infile = "y.hex"

    if "outfile" in param_dict:
        outfile = param_dict["outfile"]
    else:
        outfile = "x.hex"

    if "fformat" in param_dict:
        fformat = param_dict["fformat"]
    else:
        fformat = "hex"

    if "test_name" in param_dict:
        test_name = param_dict["test_name"]
    else:
        test_name = ""

    # select mode and do it
    if mode == "thdn":
        y = read_file(infile, fformat, nc, ns)
        for i in range(nc):
            thdn(y[i, 0:ns], fs, test_name)

    elif mode == "lpf":
        h = lpf(log2nz, log2size, nb)
        write_file(h, nb, outfile, fformat)

    elif mode == "tplot":
        y = read_file(infile, fformat, nc, ns)
        for i in range(nc):
            tplot(y[i, 0:ns], fs)

    elif mode == "fplot":
        y = read_file(infile, fformat, nc, ns)
        for i in range(nc):
            fplot(y[i, 0:ns], fs)

    elif mode == "tone":
        x = tone(fs, f, AdB, ns, nc, fdelta, flimit)
        write_file(x, nb, outfile, fformat)

    elif mode == "constant":
        x = constant(fs, f, AdB, ns, nc, fdelta)
        write_file(x, nb, outfile, fformat)

    else:
        print_usage()


if __name__ == "__main__":
    main()
