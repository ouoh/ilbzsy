import numpy as np
import math
import scipy.signal
from scipy.signal import lfilter, firwin
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
import xml.etree.ElementTree as ET
import copy
import soundfile
import resampy


"""
Zhao Shuyang, contact@zhaoshuyang.com
"""

class AudioFileProcessor:
    """
    Audio analyzer for mono-channel signal for off-line use. Picking the channel 1 for multi-channel audio file. The analyzer reads a processing document and do the required processing to the target audio.
    """
        
    def __init__(self):
        pass
        
    def load_audio(self, filename):
        #self.sig does not change along with audio processing.
        #if filename[-4:] in ['.wav', '.WAV']:
            #self.sig, self.fs, self.enc = scikits.audiolab.wavread(filename)
        #elif filename[-4:] in ['.flac', '.FLAC']:

        print(filename)
        if filename[-4:] in ['.wav', '.WAV']:
            sig, fs = soundfile.read(filename)
            
        elif filename[-5:] in ['.flac', '.FLAC']:
            sig, fs = soundfile.read(filename)
            
        sig = sig.T
                
        #Transfer to Mono
        if len(sig.shape) == 2:
            sig = sig[0]

            
        if fs != 44100:
            sig = resampy.resample(sig, fs, 44100)

        self.sig, self.duration = sig, len(sig)*1./44100

        self.fs = 44100

    def load_process_document(self, xml_file):
        self.proc_tree = ET.ElementTree()
        try:
            self.proc_tree.parse(xml_file)
        except:
            print('Not a valid XML document!')
            raise
        self.proc_root = self.proc_tree.getroot()
        if self.proc_root.tag != 'AudioAnalysis':
            print('Audio processing document is not valid!')
            raise

    def load_annotation_document(self, xml_file):
        self.ann_tree = ET.ElementTree()
        try:
            self.ann_tree.parse(xml_file)
        except:
            print('Invalid XML annotation document. Seen as pure silence.')
            raise
            self.ann_root  = ET.Element('Annotation')
            self.ann_tree._setroot(self.ann_root)
                
        self.ann_root = self.ann_tree.getroot()
        if self.ann_root.tag != 'Annotation':
            print('Audio annotation document is not valid!')
            raise
        
    def process_audio(self, verbose=False):
        """
        Recursively process audio with the tree structure parsed from the XML processing document.
        """
        if not hasattr(self, 'sig'):
            print('Audio is not loaded!')
            raise

        if not hasattr(self, 'proc_root'):
            print('Processing document is not loaded!')
            raise

        output = {}
        self.proc_root.set('result', copy.deepcopy(self.sig))

        #Construct a dictionary: {child:parent} for every node
        parent_map = {c:p for p in self.proc_root.iter() for c in p}

        for cur_node in self.proc_root.iter():

            if cur_node.tag == 'AudioNormalization':
                #Inherit the processed result and paramters from parent node
                parent_result = copy.deepcopy(parent_map[cur_node].get('result'))
                cur_node.set('result',  parent_result/np.amax(np.absolute(parent_result)))
                
            elif cur_node.tag == 'Windowing':
                #Inherit the processed result and paramters from parent node
                parent_result = copy.deepcopy(parent_map[cur_node].get('result'))
                #print len(parent_result)
                #Read new parameters
                frame_length = getSampleNum(cur_node.get('FrameLength'), self.fs)
                frame_step = getSampleNum(cur_node.get('FrameStep'), self.fs)
                #cur_node.param['FrameLength'] = frame_length
                #cur_node.param['FrameStep'] = frame_step                
                window_type = cur_node.get('Type')
                if verbose:
                    print(("[Windowing] {0}, {1}, {2}".format(frame_length, frame_step, window_type)))
                
                #Process
                cur_node.set('result', Enframe(parent_result, frame_length, frame_step, window_type))

            elif cur_node.tag == 'FFT':
                #Inherit the processed result and paramters from parent node
                parent_result = copy.deepcopy(parent_map[cur_node].get('result'))
                
                #Read new parameters
                nfft = int(cur_node.attrib.get('NFFT')) or self.fs

                #Process
                f_sig = fft(parent_result, nfft)
                spec = np.sqrt(np.real(np.conjugate(f_sig)*f_sig))
                cur_node.set('result', spec)

            elif cur_node.tag == 'Modulation4Hz':
                #Inherit the processed result and paramters from parent node
                parent_result = copy.deepcopy(parent_map[cur_node].get('result'))

                #Process
                bin_4hz_index = 4*self.fs/cur_node.param['NFFT']
                res = np.log(parent_result[:,bin_4hz_index]/np.std(parent_result, axis=1) + np.finfo(float).eps)
                cur_node.set('result', res.reshape((len(res), 1)))

                
            elif cur_node.tag == 'MelEnergy':
                #Inherit the processed result and paramters from parent node
                parent_result = copy.deepcopy(parent_map[cur_node].get('result'))

                #Read new parameters
                nfft = int(cur_node.attrib.get('NFFT')) or 1024
                Nfilter = int(cur_node.attrib.get('NFilter')) or 40
                low_freqz = cur_node.get('LowFreqz') and float(cur_node.get('LowFreqz'))/self.fs or 0
                high_freqz = cur_node.get('HighFreqz') and float(cur_node.get('HighFreqz'))/self.fs or 0.5
                                
                if verbose:
                    print(("[Calculating Mel-band Energy] Number of filters:{0}, Frequency Low-cut:{1}, Frequency high-cut:{2}".format(Nfilter, low_freqz, high_freqz)))

                #Process
                cur_node.set('result',MelSpec(parent_result, self.fs, nfft, Nfilter, low_freqz, high_freqz))

            elif cur_node.tag == 'Log':
                parent_result = copy.deepcopy(parent_map[cur_node].get('result'))
                cur_node.set('result', np.log(parent_result + np.finfo(float).eps))
                
            elif cur_node.tag == 'DCT':
                parent_result = copy.deepcopy(parent_map[cur_node].get('result'))
                cur_node.set('result', dct(parent_result, type=2, norm='ortho'))

            elif cur_node.tag == 'ExtractCoefficients': 
                parent_result = copy.deepcopy(parent_map[cur_node].get('result'))
                i1, i2 = cur_node.get('IndexRange').split('-')
                i1, i2 = int(i1), int(i2)
                cur_node.set('result', parent_result[:, i1:i2])

            elif cur_node.tag == 'Delta':
                radius = int(cur_node.get('Radius'))
                parent_result = copy.deepcopy(parent_map[cur_node].get('result'))
                cur_node.set('result', Delta(parent_result, radius))

            if cur_node.get('Output'):
                output[cur_node.get('Output')] = cur_node.get('result')

        #Unify feature length
        N_max = 0
        for k in list(output.keys()):
            if N_max < len(output[k]):
                N_max = len(output[k])
        #print 'N_max:', N_max
        for k in list(output.keys()):
            if len(output[k]) != N_max:
                n,d = output[k].shape
                repeat_n = N_max/n
                pad_n = N_max%n
                tmp_output = np.zeros((N_max,d))
                tmp_output[:N_max-pad_n, :] = np.repeat(output[k], repeat_n, axis=0)
                output[k] = tmp_output
                
        self.features = output                
        return output
            
    def process_annotation(self, verbose=False, annotation_class=False, onset=0, offset=9999):
        """
        Reqruire: Annotation and processing document to be loaded
        Return: a list of background truth for each frame
        """
        if not hasattr(self, 'proc_root'):
            print('Processing document is not loaded!')
            raise

        fs = self.proc_root.find('.//Resample') is not None and int(self.proc_root.find('.//Resample').get('SamplingRate')) or self.fs
        win_len = getSampleNum(self.proc_root.find('.//Windowing').get('FrameLength'), fs)
        w_hop = getSampleNum(self.proc_root.find('.//Windowing').get('FrameStep'), fs)
        Nframe = int(len(self.sig)/w_hop - 2)

        ground_truth = [set([]) for _ in range(Nframe)]

        if not annotation_class:
            if not hasattr(self, 'ann_root'):
                print('Annotation document is not loaded!')
                raise
        
            for segment_node in self.ann_root.findall('Segment'):
                t1 = float(segment_node.find('StartTime').text)
                t2 = float(segment_node.find('EndTime').text)
                label = segment_node.find('Label').text.strip()
                if verbose:
                    print((t1,t2, label))
                i1, i2 = time2Index(t1, t2, fs, win_len, w_hop)
                if i2 > Nframe:
                    i2 = Nframe
                for i in range(i1,i2):
                    ground_truth[i].add(label)
            return np.array(ground_truth), i1, i2
        
        else:
            if verbose:
                print((annotation_class, onset, offset))
            i1, i2 = time2Index(onset, offset, fs, win_len, w_hop)
            if i2 > Nframe:
                i2 = Nframe
            for i in range(i1,i2):
                ground_truth[i].add(annotation_class)
            #ground_truth = [set([annotation_class]) for _ in range(Nframe)]
            return np.array(ground_truth)
        
    def export_feature(self, file_name):
        np.save(file_name, self.features)

    def export_frame_labels(self, file_name):
        np.save(file_name, self.frame_labels)
        
def time2Index(t1, t2, fs, win_len, w_hop):
    #Time range to frame index
    f_begin = int(math.floor(t1*fs/w_hop))
    f_end = int((t2*fs-win_len)/w_hop)
    return (f_begin,f_end)

def Enframe(sig, win_len, w_hop, win_type):
    #win_len and w_hop in sample number
    if win_type.lower() == 'hamming':        
        win = np.hamming(win_len)
    elif win_type.lower() == 'hanning':
        win = np.hanning(win_len)
    else:
        win = np.ones(win_len)
    sig_len = len(sig)      
    Nframe = int(sig_len/w_hop - 2)
    #print Nframe, win_len

    sig_frames = np.zeros((Nframe,win_len))
    for i in range(Nframe):
        #print win_len, int(w_hop*i), int(w_hop*i) + win_len
        #print len(sig_frames[i]),  len(sig[int(w_hop*i): int(w_hop*i) + win_len]), len(win)
        
        sig_frames[i] = sig[int(w_hop*i): int(w_hop*i) + win_len] * win
    return sig_frames

def MelTrFilters(fs, nfft, Nfilter=0, low_freqz=0, high_freqz=0.5):
    if Nfilter == 0: #Default number of filter banks
        Nfilter = int(math.ceil(4.6*math.log10(fs)))
    if low_freqz==0:
        low_freqz = 25./fs #Lowest audible frequency
        
    fbank = np.zeros((Nfilter, nfft)) 
    nfreqs = np.arange(nfft) / (1. * nfft) * fs #Frequency in Hz
    
    mel_range = high_freqz - low_freqz
    mflh = fs * np.array([low_freqz, high_freqz]) 
    mflh = freq2mel(mflh) #Mapping Hz to Mel scale
    freqs = np.linspace(mflh[0], mflh[1], Nfilter + 2) #Linear band interval in Mel scale
    freqs = mel2freq(freqs) #Mapping Mel scale to Hz
    
    #Mel center frequency
    for i in range(Nfilter):
        #Creating triangular filters
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]
        l_bins = np.arange(np.floor(low * nfft / fs) + 1, np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = 1. / (cen - low)
        r_bins = np.arange(np.floor(cen * nfft / fs) + 1, np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = 1. / (hi - cen)
        fbank[i][l_bins] = lslope * (nfreqs[l_bins] - low)
        fbank[i][r_bins] = rslope * (hi - nfreqs[r_bins])        

    return fbank

def MelSpec(frames, fs, nfft, Nfilter, low_freqz, high_freqz):
    f_sig = fft(frames,nfft)
    Nframes = frames.shape[0]
    spec = np.sqrt(np.real(np.conjugate(f_sig)*f_sig))
    fbank = MelTrFilters(fs, nfft, Nfilter, low_freqz, high_freqz)
    mspec = np.dot(spec, fbank.T)
    return mspec
    
def Delta(feature_frames, n=4):
    """
    feature frames: t frames * d dimensions features
    D_t = \frac{\sum_{n=1}^N{c_{t+n} - c_{t-n}}}{2\sum_{n=1}^{N}{n^2}}
    """
    #Padding n to the start and the end
    pff = np.lib.pad(feature_frames,((n,n),(0,0)),'edge')
    norm_factor = delta_norm(n)
    delta = lfilter(-np.arange(-n,n+1), 1, pff, 0)/norm_factor
    return delta[2*n:,:]

delta_norm = lambda n: np.dot(np.power(np.arange(-n,n+1),2),  np.ones(2*n+1))
    
def freq2mel(frq):
    k = 1000./math.log(1+1000./700); #1127.01048
    mel = np.sign(frq)*np.log(1 + abs(frq)/700.)*k;
    return mel

def mel2freq(mel):
    k=1000./math.log(1+1000./700);
    frq = np.sign(mel)*700*(np.exp(abs(mel)/k)-1);
    return frq

def getSampleNum(string, fs):
    #Get frame sample number with string added with unit or not.
    if string[-2:] == 'ms':
        return int(float(string[:-2])/1000.*fs)
    elif string[-1] == 's':
        return int(float(string[:-1])*fs)
    else:
        return int(string)

