import collections
import pretty_midi
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from typing import Optional


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    """
    Extracts notes from MIDI file to pandas dataframe

    Args:
        midi_file: Path to MIDI file
    Returns:
        Pandas dataframe that contains pitch, start and end for each note
    """
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)

    for note in sorted_notes:
        notes['pitch'].append(note.pitch)
        notes['start'].append(note.start)
        notes['end'].append(note.end)
    
    return pd.DataFrame(notes)


def round_partial(value: float, resolution: Optional[float] = 1) -> float:
    """
    Rounds value to float intervals

    Args:
        value: Value to round
        resolution: Interval size (defualt = 1)
    Returns:
        Rounded value
    """
    return round(value / resolution) * resolution


def tokenize_notes(notes: pd.DataFrame, resolution: float) -> pd.DataFrame:
    """
    Converts notes to tokens of the same length

    Args:
        notes: Pandas dataframe with notes pitch, start and end
        resolution: Notes length in seconds (can be less than 1)
    Returns:
        Pandas dataframe with tokens pitch and timestamp
    """
    tokens = collections.defaultdict(list)

    for _, note in notes.iterrows():
        start = round_partial(note['start'], resolution)
        timestamp = int(start / resolution)

        tokens['timestamp'].append(timestamp)
        tokens['pitch'].append(note['pitch'])

    tokens = pd.DataFrame(tokens).sort_values('timestamp')
    tokens['pitch'] = tokens['pitch'].astype(int)
    return tokens


def tokens_to_notes(tokens: pd.DataFrame, resolution: float) -> pd.DataFrame:
    """
    Converts tokens to notes

    Args:
        tokens: Pandas dataframe with tokens pitch and timestamp
    Returns:
        Pandas dataframe with notes pitch, start and end
    """
    notes = collections.defaultdict(list)

    for i, token in tokens.iterrows():
        start = token['timestamp'] * resolution
        end = start + resolution
        
        notes['pitch'].append(token['pitch'])
        notes['start'].append(start)
        notes['end'].append(end)
    
    notes = pd.DataFrame(notes).sort_values('start')
    notes['pitch'] = notes['pitch'].astype(int)
    return notes


def notes_to_midi(
    notes: pd.DataFrame,
    instrument_name: str,
    velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:
    """
    Creates PrettyMIDI object from notes

    Args:
        notes: Pandas dataframe tokens pitch, start and end
        instrument_name: Name of the instrument that will be playing notes
        velocity: Note loudness
    returns:
        Created PrettyMIDI object
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name
        )
    )

    for _, note in notes.iterrows():
        start = note['start']
        end = note['end']
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end
        )
        instrument.notes.append(note)
    
    pm.instruments.append(instrument)
    # pm.write(out_file)
    return pm


def generate_notes(
    model: nn.Module,
    sample: torch.tensor,
    time: float,
    resolution: float,
    device: str = 'cpu',
) -> pd.DataFrame:
    """
    Generates notes dataframe using model and sample tensor

    Args:
        model: Torch RNN model
        sample: Sample tensor of shape (n, 128)
        time: Length of generated sequence in seconds
        resolution: Resolution of notes tokenization
        device: Torch device
    Returns:
        Generated notes dataframe
    """
    model.eval()

    # Process sample notes
    sample = sample.to(device)
    hidden = model.init_hidden(1).to(device)
    for i in range(sample.shape[0]):
        output, hidden = model(
            sample[i].unsqueeze(0).unsqueeze(0),
            hidden,
        )
    
    # Generate notes
    input = output
    outputs = []
    for i in range(int(time / resolution - sample.shape[0])):
        output, hidden = model(input, hidden)

        input = torch.round(torch.sigmoid(output))
        outputs.append(input.detach().cpu().numpy())
    
    # Join outputs
    outputs = np.concatenate(outputs, axis=1)[0]
    # Join sample and generated notes
    outputs = np.concatenate(
        (sample.detach().cpu().numpy(), outputs),
        axis=0,
    )

    # Convert model output to tokens
    tokens = collections.defaultdict(list)
    for timestamp, pitch in zip(*np.where(outputs)):
        tokens['timestamp'].append(timestamp)
        tokens['pitch'].append(pitch)
    tokens = pd.DataFrame(tokens)

    # Convert tokens to notes
    notes = tokens_to_notes(tokens, resolution)

    return notes
