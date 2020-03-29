  # DT2119, Lab 1 Feature Extraction
import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as signal
from scipy.fftpack import fft
from scipy.fftpack import dct
import lab1_tools as lab1tools
# Function given by the exercise ----------------------------------


def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lab1tools.lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    N = samples.shape[0]
    output = np.array(samples[0:winlen])
    i = winshift
    for i in range (winshift, N-winlen, winshift) :
        output = np.vstack((output,samples[i:i+winlen]))
    return output  

    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    b = np.array([1, -p])
    a = np.array([1])
    output = signal.lfilter(b,a,input,axis=1)# high pass FIR filter y(t)=x(t)-p*x(t-1) 
    return output 


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    N, M = np.shape(input)
    output =input*signal.hamming(M, sym=0)
    return output


def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    result = np.abs(fft(input, nfft))**2
    return result

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    fbank = lab1tools.trfbank(samplingrate, input.shape[1])
    return np.log(np.dot(input,fbank.T))


def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    return dct(input)[:, :nceps]

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """

if __name__ == "__main__":
    data = np.load('lab1_data.npz',allow_pickle=True)['data']
    example = np.load('lab1_example.npz',allow_pickle=True)['example'].item()
    samples = example['samples']
    samplingrate = int(example['samplingrate']/1000)  #sampling per millisecond
    winlen = 20*samplingrate
    winshift = 10*samplingrate

    t=7
    fig, ax = plt.subplots(nrows=t,ncols=2,figsize=(t+3,t*2))
    emph = enframe(samples,winlen,winshift)  
    ax[0][0].pcolormesh(emph.T)
    ax[0][0].set_title('enframed samples')
    ax[0][1].pcolormesh(example['frames'].T)

    preemp_emph = preemp(emph)
    ax[1][0].pcolormesh(preemp_emph.T)
    ax[1][0].set_title('preemphasis')
    ax[1][1].pcolormesh(example['preemph'].T)

    windows = windowing(preemp_emph)
    ax[2][0].pcolormesh(windows.T)
    ax[2][0].set_title('hamming window')
    ax[2][1].pcolormesh(example['windowed'].T)

    pSpec = powerSpectrum(windows,512)
    ax[3][0].pcolormesh(pSpec.T)
    ax[3][0].set_title('abs(FFT)$^2$')
    ax[3][1].pcolormesh(example['spec'].T)

    melSpec = logMelSpectrum(pSpec,20000)
    ax[4][0].pcolor(melSpec.T)
    ax[4][0].set_title('Mel')
    ax[4][1].pcolormesh(example['mspec'].T)

    cStrum = cepstrum(melSpec,13)
    ax[5][0].pcolormesh(cStrum.T)
    ax[5][0].set_title('mfcc')
    ax[5][1].pcolormesh(example['mfcc'].T)

    ax[6][0].pcolormesh(lab1tools.lifter(cStrum).T)
    ax[6][0].set_title('lmfcc')
    ax[6][1].pcolormesh(example['lmfcc'].T)

    fig.tight_layout()
    plt.savefig("results_of_examples.png")
    plt.show()

    #feature correlation
    mfccFeatures = mfcc(data[0]['samples'])
    mspecFeatures = mspec(data[0]['samples'])

    for i in range(1,len(data)):
        mfccFeatures = np.vstack((mfccFeatures,mfcc(data[i]['samples'])))
        mspecFeatures = np.vstack((mspecFeatures,mspec(data[i]['samples'])))

    mfccCorr = np.corrcoef(mfccFeatures.T)
    plt.title("MFCC Features correlations")
    plt.pcolormesh(mfccCorr)
    plt.show()


    mspecCorr = np.corrcoef(mspecFeatures.T)
    plt.title("MSPEC Features correlations")
    plt.pcolormesh(mspecCorr)
    plt.show()


