import glob
import os
import librosa
import numpy as np
from hparam import hparam as hp
from pyAudioAnalysis import audioSegmentation as aS
import python_speech_features as psf
import soundfile as sf
from multiprocessing import Pool

# This preprocess method is a modification from Harry Volek's original implementation
# The features extracted are from python_speech_features library and the array shapes are different
# The silence detection is done from a custom function which in turn uses the library pyAudioAnalysis


# downloaded dataset path
audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))

def silence_detection(utter):
    intervals = aS.silence_removal(utter, 
                                   hp.data.sr, 
                                   0.025, 
                                   0.010, 
                                   smooth_window = 1.0, 
                                   weight = 0.3, 
                                   plot = False)

    for i in range(len(intervals)):
        intervals[i] = [int(stamp*hp.data.sr) for stamp in intervals[i]]

    return intervals

# downloaded dataset path
audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))

def silence_detection(utter):
    intervals = aS.silence_removal(utter, 
                                   hp.data.sr, 
                                   0.025, 
                                   0.010, 
                                   smooth_window = 1.0, 
                                   weight = 0.3, 
                                   plot = False)

    for i in range(len(intervals)):
        intervals[i] = [int(stamp*hp.data.sr) for stamp in intervals[i]]

    return intervals

def extract_speakerwise_spec(speaker,test_dataset_path = hp.data.test_path_unprocessed[:-10],single_utterance=True,task='test'):
    
    if not(os.path.exits(hp.data.test_path)):
        os.makedirs(hp.data.test_path)
        
    speaker_path = os.path.join(test_dataset_path,speaker)
    sessions_spec = []

    for session in os.listdir(speaker_path):
        if single_utterance and len(sessions_spec)>0: # for miniworkflow break as soon as a spectrogram from a single utterance is obtained ; no need to traverse full dataset
            break
        session_path = os.path.join(speaker_path,session)

        utterances_spec = []
        for utter_name in os.listdir(session_path):
            
            if single_utterance and len(sessions_spec)>0:
                break

            if utter_name[-4:] == '.wav':

                utter_path = os.path.join(session_path, utter_name)         # path of each utterance
                utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio

                intervals = silence_detection(utter)

                bits_per_sample = sf.SoundFile(utter_path)
                bits_per_sample = int(bits_per_sample.subtype[-2:])

                utter /= 2**(bits_per_sample-1)

                utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length

                for interval in intervals:
                    if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                        utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                        S = psf.base.logfbank(utter_part, 
                                             samplerate=hp.data.sr, 
                                             winlen=hp.data.window,
                                             winstep=hp.data.hop, 
                                             nfilt=hp.data.nmels, 
                                             nfft=hp.data.nfft, 
                                             lowfreq=0, 
                                             highfreq=None, 
                                             preemph=0.97)
                        utterances_spec.append(S)

        if len(utterances_spec) > 0:
            if len(sessions_spec) > 0:
                sessions_spec = np.vstack((sessions_spec,np.vstack(utterances_spec)))            
                
            else:
                sessions_spec = np.vstack(utterances_spec)

        print(speaker," : ",sessions_spec.shape)
        if task =='train':      # save spectrogram as numpy file
            np.save(os.path.join(hp.data.train_path, f'{speaker}.npy'), sessions_spec)
        elif task == 'test':
                np.save(os.path.join(hp.data.test_path, f'{speaker}.npy'), sessions_spec)


if __name__ == "__main__":
    speaker_list = os.listdir(hp.data.test_path_unprocessed[:-10])
    Pool().map(extract_speakerwise_spec,speaker_list)
