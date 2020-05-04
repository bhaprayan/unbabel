import glob

import pandas as pd
import torchaudio
from torch.utils.data import Dataset

import os

class AccentTransferDataset(Dataset):
    def __init__(self):
        dat = pd.read_csv("speakers_all.csv")
        self.usa_english_males = dat.loc[
            (dat["file_missing?"] is False)
            & (dat["country"] == "usa")
            & (dat["native_language"] == "english")
            & (dat["sex"] == "male")
        ]
        self.uk_english_males = dat.loc[
            (dat["file_missing?"] is False)
            & (dat["country"] == "usa")
            & (dat["native_language"] == "english")
            & (dat["sex"] == "male")
        ]   


class PollyDataset(Dataset):
    def __init__(self):
        uk_words = set(map(lambda x: x.split("/")[-1][:-7], glob.glob("data/uk/*")))
        us_words = set(map(lambda x: x.split("/")[-1][:-7], glob.glob("data/us/*")))
        common_words = list(uk_words.intersection(us_words))
        self.uk_words_common = list(
            map(lambda x: "data/uk/" + x + "_uk.mp3", common_words)
        )
        self.us_words_common = list(
            map(lambda x: "data/us/" + x + "_us.mp3", common_words)
        )

        self.sg_transform = torchaudio.transforms.Spectrogram()

    def __getitem__(self, idx):
        # load and compute spectrogram (sg) for us accent word
        us_audio = torchaudio.load(self.us_words_common[idx])[0]
        us_sg = F.interpolate(self.sg_transform(us_audio), size=128)
        # load and compute spectrogram (sg) for uk accent word
        uk_audio = torchaudio.load(self.uk_words_common[idx])[0]
        uk_sg = F.interpolate(self.sg_transform(uk_audio), size=128)
        print(self.us_words_common[idx], self.uk_words_common[idx])
        return (us_sg, uk_sg)

    def __len__(self):
        return len(self.us_words_common)
