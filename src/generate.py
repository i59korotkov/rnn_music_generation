import os
import yaml
import torch
import matplotlib.pyplot as plt

from src.train import plot_piano_roll
from src.models import RNN
from src.utils.midi import midi_to_notes, tokenize_notes, generate_notes, notes_to_midi
from src.utils.data import NotesDataset, PITCH_CNT

CONFIG_PATH = 'config/generate.yaml'


if __name__ == '__main__':
    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    device = config['device']

    # Create model
    print('Loading model...')
    model = RNN(PITCH_CNT, config['model']['hidden_size'], config['model']['num_layers']).to(device)
    model.load_state_dict(torch.load(config['model_weights_path']))

    # Parse sample file
    print('Parsing sample file...')
    notes = midi_to_notes(config['sample_file_path'])
    tokens = tokenize_notes(notes, config['resolution'])
    tokens['filename'] = config['sample_file_path']
    sample_tensor = NotesDataset(tokens)[0][0]
    sample_tensor_start = NotesDataset(tokens)[0][0][:config['sample_timestamps_cnt']]

    # Generate notes
    print('Generating notes...')
    generated_notes = generate_notes(
        model,
        sample_tensor_start,
        config['length'],
        config['resolution'],
        device,
    )

    # Create output folder
    output_folder_path = config['output_folder_path']
    os.makedirs(output_folder_path)

    # Generate MIDI file
    out_file = notes_to_midi(generated_notes, config['instrument_name'])
    out_file.write(os.path.join(output_folder_path, 'generated_midi.midi'))

    # Generate piano rolls
    sample_notes = generate_notes(
        model,
        sample_tensor,
        config['length'],
        config['resolution'],
        device,
    )
    plot_piano_roll(sample_notes, time=config['length'])
    plt.savefig(os.path.join(output_folder_path, 'sample_piano_roll.png'))

    plot_piano_roll(generated_notes, time=config['length'])
    plt.savefig(os.path.join(output_folder_path, 'generated_piano_roll.png'))

    print('Output files generated.')
