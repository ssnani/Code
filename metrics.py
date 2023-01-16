import numpy as np
from pystoi import stoi
import pesq
import pypesq
import torch 
import pytorch_lightning
from multiprocessing import Pool, set_start_method, get_context
#set_start_method("spawn")




class spectrum_analysis(object):
    def __init__(self):
        self.window_len = 512
        self.hop_size = 128 #64
        self.window_callback = torch.hamming_window
        self.sr = 16000

_spectrum_config = spectrum_analysis()


def _mag_spec_mask(mask, mag):
    masked_mag = mask*mag
    return masked_mag

def gettdsignal_mag_spec_mask(mask, mag, ph):
    masked_mag = mask*mag
    # Shape of the istft((*, freq, time))
    if (mag.shape[-1] == _spectrum_config.window_len//2 + 1): #freq axis check
        batch_size, time_idx, freq_idx = mag.shape 
        masked_mag = torch.permute(masked_mag, [0,2,1]) #torch.reshape(masked_mag, (batch_size, freq_idx, time_idx ))
        ph = torch.permute(ph, [0,2,1])#torch.reshape(ph, (batch_size,  freq_idx, time_idx ))
    
    _est_cs = torch.polar(masked_mag, ph)
    hamming_window = torch.hamming_window(_spectrum_config.window_len).type_as(masked_mag)
    sig = torch.istft(_est_cs, _spectrum_config.window_len, _spectrum_config.hop_size, _spectrum_config.window_len, hamming_window)
    return sig

def _my_istft(cs_spec):
    # Shape of the istft((*, freq, time))
    if (cs_spec.shape[-1] == _spectrum_config.window_len//2 + 1): #freq axis check
        batch_size, time_idx, freq_idx = cs_spec.shape 
        cs_spec = torch.permute(cs_spec, [0,2,1]) 
    
    hamming_window = torch.hamming_window(_spectrum_config.window_len)
    sig = torch.istft(cs_spec, _spectrum_config.window_len, _spectrum_config.hop_size, _spectrum_config.window_len, hamming_window)
    return sig

def si_snr(tgt, est):
    s_tgt = np.sum(est*tgt, axis=-1, keepdims=True)*tgt / \
    np.sum(tgt*tgt, axis=-1, keepdims=True)
    e_ns = est - s_tgt
    ans = 10.*np.log10(np.sum(s_tgt*s_tgt, axis=-1)/np.sum(e_ns*e_ns, axis=-1))
    return ans

def snr(s, s_p):
    r""" calculate signal-to-noise ratio (SNR)

        Parameters
        ----------
        s: clean speech
        s_p: processed speech
    """
    return 10.0 * np.log10(np.sum(s**2)/np.sum((s_p-s)**2))

def PESQ(ref_wav, deg_wav):
    rate = 16000
    return pesq.pesq(rate, ref_wav, deg_wav, 'wb', on_error = 1)

def eval_metrics_v1(ref_wav, est_wav): #inputs

    #ref_wav, est_wav = inputs
    srate = 16000

    #breakpoint()

    ref_wav = ref_wav.reshape(-1)
    est_wav = est_wav.reshape(-1)
    #print(".......... Evaluting Metrics .........")
    _snr = snr(ref_wav, est_wav)
    _si_snr = si_snr(ref_wav, est_wav)


    #try:
    pesq_wb = PESQ(ref_wav, est_wav)
    
    pesq_nb = pypesq.pesq(ref_wav, est_wav, srate) #original   #TODO: installation error on delta erver
    #pesq_nb = -1.0
    
    #except PesqError.NO_UTTERANCES_DETECTED:#Pesq.NoUtterancesError:
    #    print("***** No Utterance Error ****")
    #pesq_wb = -100
    #    pesq_nb = -100  #Invalid
    
    _stoi = stoi(ref_wav, est_wav, srate, extended=False)
    e_stoi = stoi(ref_wav, est_wav, srate, extended=True)

    return {'snr': _snr, 'si_snr': _si_snr, 'pesq_nb': pesq_nb, 'pesq_wb': pesq_wb, 'stoi': _stoi, 'e_stoi': e_stoi}

def eval_metrics_batch_v1(ref_wavs, est_wavs):
    #expected shape: (batch_size, num_samples)
    accu_metrics = [0, 0, 0, 0, 0, 0]
    for batch in range(ref_wavs.shape[0]):
        est_d = eval_metrics_v1(ref_wavs[batch,:].reshape(-1),est_wavs[batch,:].reshape(-1))
        
        #print(est_d)
        _snr = est_d['snr']
        _si_snr = est_d['si_snr']
        pesq_nb = est_d['pesq_nb']
        pesq_wb = est_d['pesq_wb']
        _stoi = est_d['stoi']
        e_stoi = est_d['e_stoi']

        accu_metrics[0] += _snr
        accu_metrics[1] += _si_snr
        accu_metrics[2] += pesq_nb
        accu_metrics[3] += pesq_wb
        accu_metrics[4] += _stoi
        accu_metrics[5] += e_stoi

    avg_acc_metrics = np.array(accu_metrics)/(ref_wavs.shape[0]+1e-8)
    avg_snr, avg_si_snr, avg_pesq_nb, avg_pesq_wb, avg_stoi, avg_e_stoi = avg_acc_metrics

    return {'snr': avg_snr, 'si_snr': avg_si_snr, 'pesq_nb': avg_pesq_nb, 'pesq_wb': avg_pesq_wb, 'stoi': avg_stoi, 'e_stoi': avg_e_stoi}


def eval_metrics(inputs): #

    ref_wav, est_wav = inputs
    srate = 16000

    #breakpoint()

    ref_wav = ref_wav.reshape(-1)
    est_wav = est_wav.reshape(-1)
    #print(".......... Evaluting Metrics .........")
    _snr = snr(ref_wav, est_wav)
    _si_snr = si_snr(ref_wav, est_wav)


    #try:
    pesq_wb = PESQ(ref_wav, est_wav)
    
    pesq_nb = pypesq.pesq(ref_wav, est_wav, srate) #original   #TODO: installation error on delta erver
    #pesq_nb = -1.0
    
    #except PesqError.NO_UTTERANCES_DETECTED:#Pesq.NoUtterancesError:
    #    print("***** No Utterance Error ****")
    #pesq_wb = -100
    #    pesq_nb = -100  #Invalid
    
    _stoi = stoi(ref_wav, est_wav, srate, extended=False)
    e_stoi = stoi(ref_wav, est_wav, srate, extended=True)

    return {'snr': _snr, 'si_snr': _si_snr, 'pesq_nb': pesq_nb, 'pesq_wb': pesq_wb, 'stoi': _stoi, 'e_stoi': e_stoi}

def eval_metrics_batch(ref_wavs, est_wavs):
    #expected shape: (batch_size, num_samples)
    accu_metrics = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    results = []
    inputs = [(ref_wavs[batch,:].reshape(-1), est_wavs[batch,:].reshape(-1)) for batch in range(ref_wavs.shape[0])]    
    #breakpoint()
    pool = get_context("spawn").Pool(8) #
    results = pool.map(eval_metrics, inputs)

    accu_metrics = np.array(accu_metrics)
    for result in results:
        accu_metrics += np.array(list(result.values()))

    avg_acc_metrics = accu_metrics/(ref_wavs.shape[0]+1e-8)
    #print(avg_acc_metrics)
    avg_snr, avg_si_snr, avg_pesq_nb, avg_pesq_wb, avg_stoi, avg_e_stoi = avg_acc_metrics

    return {'snr': avg_snr, 'si_snr': avg_si_snr, 'pesq_nb': avg_pesq_nb, 'pesq_wb': avg_pesq_wb, 'stoi': avg_stoi, 'e_stoi': avg_e_stoi}



def update_eval_metric(*a):

    est_d = a[0]

    _snr = est_d['snr']
    _si_snr = est_d['si_snr']
    pesq_nb = est_d['pesq_nb']
    pesq_wb = est_d['pesq_wb']
    _stoi = est_d['stoi']
    e_stoi = est_d['e_stoi']

    accu_metrics[0] += _snr
    accu_metrics[1] += _si_snr
    accu_metrics[2] += pesq_nb
    accu_metrics[3] += pesq_wb
    accu_metrics[4] += _stoi
    accu_metrics[5] += e_stoi

if __name__ == "__main__":
    x = np.random.rand(128,16000)
    n = np.random.rand(128,16000)
    y = x+n
    #y = np.zeros((16000,1))

    #x = x.reshape(-1)
    #y = y.reshape(-1)
    breakpoint()
    est_d = eval_metrics_batch(x,y)
    #breakpoint()
    print(est_d)
