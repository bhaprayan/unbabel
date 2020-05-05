import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio.datasets
import torchaudio.transforms
import tqdm

##
import matplotlib.pyplot as plt
import pdb



class PollyClassificationDataset():
    def __init__(self, recordingsPath, accents = ['uk', 'us'], transform = None):
        
        self.audioFileNames = []
        self.audioFileLabels = []

        for accent in accents:
            accentPath = os.path.join(recordingsPath, accent)
            for fn in os.listdir(accentPath):
                file_location = os.path.join(accentPath, fn)
                self.audioFileNames.append(file_location)
                self.audioFileLabels.append(accent)
        
        indexes = list(range(len(set(self.audioFileLabels))))
        label_names = sorted(set(self.audioFileLabels))

        self.idx_to_labels = { pair[0]:pair[1] for pair in zip(indexes, label_names) }
        self.labels_to_idx = { v:k for k,v in self.idx_to_labels.items() }
        self.audioFileLabels = list(map(lambda x: self.labels_to_idx[x], self.audioFileLabels))
     
        # get class counts
        self.class_counts = {}
        for i in indexes:
            self.class_counts[self.idx_to_labels[i]] = self.audioFileLabels.count(i)
        
        
        if transform is None:
            window_len   = 0.025 # in ms
            window_shift = 0.01 # in ms
            samplerate   = 16e3
            n_fft        = int(samplerate * window_len)
            hop_len      = int(samplerate * window_shift)
            transform = torchaudio.transforms.Spectrogram(n_fft = n_fft, hop_length = hop_len,
                                                          pad = 10, normalized = True)
        
        self.transform = transform
        print("Tranform applied to data is:", self.transform)
    
    def __getitem__(self, idx):
        fn = self.audioFileNames[idx]
        
        try: 
            audio, samplerate = torchaudio.load(fn, normalization = False) # don't normalize audio
        except:
            print("CAUGHT EXCEPTION: couldn't open file" + fn + '. Going to return empty tensor')
            samplerate = 22050
            audio = torch.zeros(40000).view(1, -1)
        if (samplerate != 22050):
            raise Exception("Input file sample rate is {}, expected 22050".format(samplerate))

        num_elem_wanted = 40000 # keep 79787 elems so spectrogram will have len 500 which covers most examples
        
        if audio.numel() <= num_elem_wanted:
            # pad the input on both sides
            pad_size = (num_elem_wanted-audio.numel())//2
            # we need to account for off by 1 errors due to integer division --> pad by small number to avoid numerical error
            if 2*pad_size + audio.numel() == num_elem_wanted:
                audio = torch.constant_pad_nd(audio, pad = (pad_size,pad_size), value=0)
            else:
                audio = torch.constant_pad_nd(audio, pad = (pad_size,pad_size+1), value=0)
        else:
            # slice input in the middle
            start = audio.shape[1]//2 - num_elem_wanted//2
            end = start + num_elem_wanted
            audio = audio[:, start:end]

        # pdb.set_trace()
        # assert((audio.shape[0],audio.shape[1]) == (1, num_elem_wanted))
        if self.transform is not None:
            audio = self.transform(audio[0])
            
        SMALL_CONSTANT = 1e-5
        audio = audio + SMALL_CONSTANT # to avoid numerical errors
        audio = audio.log2()


        
        speaker = self.audioFileLabels[idx]
        audio = audio.unsqueeze(0)
        return audio, speaker

    def __len__(self):
        return len(self.audioFileNames)
    
    def getAudio(self, idx):
        fn = self.audioFileNames[idx]
        try: 
            audio, samplerate = torchaudio.load(fn, normalization = False) # don't normalize audio
        except:
            print("CAUGHT EXCEPTION: couldn't open file" + fn + '. Going to return empty tensor')
            samplerate = 22050
            audio = torch.zeros(40000).view(1, -1)
        if (samplerate != 22050):
            raise Exception("Input file sample rate is {}, expected 22050".format(samplerate))

        num_elem_wanted = 40000 # keep 79787 elems so spectrogram will have len 500 which covers most examples
        
        if audio.numel() <= num_elem_wanted:
            # pad the input on both sides
            pad_size = (num_elem_wanted-audio.numel())//2
            # we need to account for off by 1 errors due to integer division --> pad by small number to avoid numerical error
            if 2*pad_size + audio.numel() == num_elem_wanted:
                audio = torch.constant_pad_nd(audio, pad = (pad_size,pad_size), value=0)
            else:
                audio = torch.constant_pad_nd(audio, pad = (pad_size,pad_size+1), value=0)
        else:
            # slice input in the middle
            start = audio.shape[1]//2 - num_elem_wanted//2
            end = start + num_elem_wanted
            audio = audio[:, start:end]

        # pdb.set_trace()
        # assert((audio.shape[0],audio.shape[1]) == (1, num_elem_wanted))

        
#         print("NOT CREATING SPECTROGRAM - RETURNING ORIGINAL AUDIO")
#         if self.transform is not None:
#             audio = self.transform(audio[0])
        
        return audio, self.audioFileLabels[idx]
    '''
        SMALL_CONSTANT = 1e-5
        audio = audio + SMALL_CONSTANT # to avoid numerical errors
        audio = audio.log2()


        
        speaker = self.audioFileLabels[idx]
        audio = audio.unsqueeze(0)
        return audio, speaker
    '''




class CMUArctic(Dataset):
    def __init__(self, recordingsPath, speakers, transform = None):
        '''
        Create a dataset from list of supplied speakers in the given path

        recordingsPath: path of folder containing all folders corresponding to different speakers
        speakers: list of speakers to choose, ex: ["aew", "clb"]
        transform: 
        '''

        self.audio_files = []
        self.audio_labels = []

        for speaker in tqdm.tqdm(speakers):
            speakerPath = "cmu_us_{}_arctic".format(speaker)
            speakerPath = os.path.join(recordingsPath, speakerPath, "wav")

            for fn in os.listdir(speakerPath):
                fpath = os.path.join(speakerPath, fn)
                self.audio_files.append(fpath)
                self.audio_labels.append(speaker)
        
        indexes = list(range(len(set(self.audio_labels))))
        label_names = sorted(set(self.audio_labels))
        self.idx_to_labels = { pair[0]:pair[1] for pair in zip(indexes, label_names) }
        self.labels_to_idx = { v:k for k,v in self.idx_to_labels.items() }
        self.audio_labels = list(map(lambda x: self.labels_to_idx[x], self.audio_labels))
        
        # get class counts
        self.class_counts = {}
        for i in indexes:
            self.class_counts[self.idx_to_labels[i]] = self.audio_labels.count(i)
        
        
        if transform is None:
            window_len   = 0.025 # in ms
            window_shift = 0.01 # in ms
            samplerate   = 16e3
            n_fft        = int(samplerate * window_len)
            hop_len      = int(samplerate * window_shift)
            transform = torchaudio.transforms.Spectrogram(n_fft = n_fft, hop_length = hop_len,
                                                          pad = 10, normalized = True)
        
        self.transform = transform
        print("Tranform applied to data is:", self.transform)
        
        



    
    def __getitem__(self, idx):
        fn = self.audio_files[idx]
        
#         pdb.set_trace()
        
        audio, samplerate = torchaudio.load(fn, normalization = False) # don't normalize audio
        if (samplerate != 16e3):
            raise Exception("Input file sample rate is {}, expected 16000".format(samplerate))
        
        num_elem_wanted = 79900 # keep 79787 elems so spectrogram will have len 500 which covers most examples
        
        if audio.numel() <= num_elem_wanted:
            # pad the input on both sides
            pad_size = (num_elem_wanted-audio.numel())//2
            # we need to account for off by 1 errors due to integer division --> pad by small number to avoid numerical error
            if 2*pad_size + audio.numel() == num_elem_wanted:
                audio = torch.constant_pad_nd(audio, pad = (pad_size,pad_size), value=0)
            else:
                audio = torch.constant_pad_nd(audio, pad = (pad_size,pad_size+1), value=0)
        else:
            # slice input in the middle
            start = audio.shape[1]//2 - num_elem_wanted//2
            end = start + num_elem_wanted
            audio = audio[:, start:end]

        assert((audio.shape[0],audio.shape[1]) == (1, num_elem_wanted))
        if self.transform is not None:
            audio = self.transform(audio[0])
            
        SMALL_CONSTANT = 1e-5
        audio = audio + SMALL_CONSTANT # to avoid numerical errors
        audio = audio.log2()


        
        speaker = self.audio_labels[idx]
        audio = audio.unsqueeze(0)
        return audio, speaker
    
    def __len__(self):
        return len(self.audio_labels)

    
    
    
    
def get_training_data():
    recordingsPath = "./Arctic Data/training_data/"
    all_speakers = ['aew' , 'ahw', 'aup', 'awb', 'axb', 'bdl',
                    'clb', 'eey', 'fem', 'gka', 'jmk', 'ksp',
                    'ljm', 'lnh', 'rms', 'rxr', 'slp', 'slt']
    selected_speakers = ['aew' , 'ahw', 'aup']
    dataset = CMUArctic(recordingsPath, all_speakers)
    return dataset
    
def get_test_data():
    recordingsPath = "./Arctic Data/testing_data/"
    all_speakers = ['aew' , 'ahw', 'aup', 'awb', 'axb', 'bdl',
                    'clb', 'eey', 'fem', 'gka', 'jmk', 'ksp',
                    'ljm', 'lnh', 'rms', 'rxr', 'slp', 'slt']
    selected_speakers = ['aew' , 'ahw', 'aup']
    dataset = CMUArctic(recordingsPath, all_speakers)
    return dataset
    
    
def get_item_shapes(dataset):
    result = []
    for i in tqdm.tqdm(range(len(dataset))):
        audio, _ = dataset[i]
        result.append(audio.shape)
    return result


def display_signal_and_spectrogram(signal):
    '''
    signals: a spectrogram/transformed audio signal to display
    '''
    plt.figure()
    plt.imshow(signal)

    

if __name__ == "__main__":
    recordingsPath = "./Arctic Data/"
    speakers = ['aew'] #, 'ahw', 'aup', 'awb', 'axb', 'bdl']
                # 'clb', 'eey', 'fem', 'gka', 'jmk', 'ksp',
                #  'ljm', 'lnh', 'rms', 'rxr', 'slp', 'slt']
    
    ds = CMUArctic(recordingsPath, speakers)
    print("prepared dataset... getting audio shapes")
    all_audio_shapes = get_item_shapes(ds)
    random_signal = ds[5][0]
    
    sizes_list = list(map(lambda x: x[1], all_audio_shapes))
    pdb.set_trace()
    display_signal_and_spectrogram(random_signal)
    print("done")


