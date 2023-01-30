import yaml
import pathlib
import pandas as pd
from tqdm import tqdm

from src.utils.midi import midi_to_notes, tokenize_notes

DATA_PREPROCESSING_CONFIG_PATH = 'config/data_preprocessing.yaml'


if __name__ == '__main__':
    # Load config
    with open(DATA_PREPROCESSING_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Get all MIDI files from dataset folder
    filenames = list(map(str, pathlib.Path(config['dataset_path']).rglob('*.mid*')))
    print(f'{len(filenames)} MIDI files found in dataset folder.')

    # Parse MIDI files
    train_size = config['train_size']
    resolution = config['resolution']

    # Parse train tokens
    train_tokens = []
    for f in tqdm(filenames[:int(len(filenames) * train_size)], desc='Parsing tokens from MIDI files'):
        notes = midi_to_notes(f)
        tokens = tokenize_notes(notes, resolution)
        tokens['filename'] = f
        train_tokens.append(tokens)
    train_tokens = pd.concat(train_tokens)
    train_tokens = train_tokens.reset_index(drop=True)

    # Parse val tokens
    val_tokens = []
    for f in tqdm(filenames[int(len(filenames) * train_size):], desc='Parsing tokens from MIDI files'):
        notes = midi_to_notes(f)
        tokens = tokenize_notes(notes, resolution)
        tokens['filename'] = f
        val_tokens.append(tokens)
    val_tokens = pd.concat(val_tokens)
    val_tokens = val_tokens.reset_index(drop=True)

    # Save dataframes to csv files
    train_tokens.to_csv(config['train_tokens_path'])
    val_tokens.to_csv(config['val_tokens_path'])
