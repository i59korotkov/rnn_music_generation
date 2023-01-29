import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PITCH_CNT = 128


class NotesDataset(Dataset):
    def __init__(self, tokens: pd.DataFrame) -> None:
        self.filenames = sorted(tokens['filename'].unique())
        self.all_tokens = tokens
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        tokens = self.all_tokens[self.all_tokens['filename'] == filename]
        
        input_seq = np.zeros((tokens['timestamp'].max() + 1, PITCH_CNT), dtype='float32')
        input_seq[tokens['timestamp'].values, tokens['pitch'].values] = 1
        input_seq = torch.from_numpy(input_seq)

        output_seq = np.zeros((tokens['timestamp'].max() + 1, PITCH_CNT), dtype='float32')
        output_seq[tokens['timestamp'].values - 1, tokens['pitch'].values] = 1
        output_seq = torch.from_numpy(output_seq)

        return input_seq, output_seq
