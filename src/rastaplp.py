import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from utils import audspec,postaud,dolpc,lpc2cep,spec2cep,lpc2spec,lifter,power_spectrum

def powspec(x, sr,wintime=0.025, steptime= 0.010, dither=1):

    winpts = int(np.round(wintime*sr));
    steppts =int(np.round(steptime*sr));
    NFFT = 2**(np.ceil(np.log(winpts)/np.log(2)));
    WINDOW = np.hanning(winpts)
    NOVERLAP = winpts - steppts;
    SAMPRATE = sr;
    f,t,spec=scipy.signal.spectrogram(x*32768,1.0,WINDOW,winpts,NOVERLAP,NFFT,scaling = 'spectrum' )    
    y=spec
    if dither:
        y = y + winpts
    e = np.log(np.sum(y,0))
    return y,e
    



def rastaplp(signal, sr = 16000, dorasta=0, modelorder = 8):
    # powerspectrum
    #p_spectrum,e = powspec(signal, sr)
    p_spectrum =power_spectrum(signal*32768,fs=sr,win_time=0.025,shift=0.01,prefac=0.97)    
    p_spectrum =p_spectrum + 200;

    # group powerspectrum to critical band    
    a_spectrum=audspec(p_spectrum, fs=sr)
        
    nbands = len(a_spectrum[0])
    
    if not dorasta ==0:
        # put in log domain
        nl_a_spectrum = np.log(a_spectrum)
        # do rasta filtering
        ras_nl_a_spectrum = rastafilt(nl_a_spectrum)
        # do inverse log
        a_spectrum = np.exp(ras_nl_a_spectrum)
 
    # do final auditory compressions
    post_spectrum,eql = postaud(a_spectrum, sr/2) # it using sr/2 instead of sr, said ==> 2012-09-03 bug: was sr
 
    if modelorder > 0:
        # lpc analysis
        lpc_anal = dolpc(post_spectrum, modelorder)
        # convert lpc to cepstra
        cepstra = lpc2cep(lpc_anal, modelorder + 1)
        # or convert lpc to spectra
        spectra, F, M = lpc2spec(lpc_anal, nbands)
    else:
        # no lpc smoothing of spectrum
        spectra = post_spectrum
        cepstra = spec2cep(spectra)
 
    cepstra = lifter(cepstra, 0.6)
 
    return cepstra, spectra, p_spectrum, lpc_anal, F, M
 
def lifter_own(x, lift = 0.6, invs = 0):
    n_cep, nfrm = x.shape
 
    if lift == 0:
        y = x
    else:
        if lift > 0:
            if lift > 10:
                print("unlikely lift exponent of {} (did you mean -ve?)".format(lift))
 
            lift_wts = [1, ([1])]