import pathlib
import pandas as pd
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from src.utils.midi import midi_to_notes, tokenize_notes, tokens_to_notes, generate_notes
from src.utils.data import NotesDataset, PITCH_CNT
from src.models import RNN

TRAIN_CONFIG_PATH = 'config/train.yaml'


def train_epoch(
    model,
    loss,
    optimizer,
    train_loader,
    device: str='cpu',
    scheduler=None,
    epoch: int=0,
    writer=None,
) -> float:
    model.train()
    train_loss = 0
    for i, (input_seq, target_seq) in enumerate(tqdm(
        train_loader, total=len(train_loader), desc='Train: ', leave=False
    )):
        # Move hidden state, input and output sequences to device
        hidden = model.init_hidden(1).to(device)
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)

        output_seq, hidden = model(input_seq, hidden)
        loss_value = loss(output_seq, target_seq)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.item()

        # Write training loss to tensorobard
        if writer is not None:
            writer.add_scalar(
                'training loss',
                loss_value.item(),
                epoch * len(train_loader) + i,
            )

    train_loss /= len(train_loader)

    if scheduler is not None:
        scheduler.step()
    
    return train_loss


def validate_model(
    model,
    loss,
    val_loader,
    device: str='cpu',
) -> float:
    model.eval()
    val_loss = 0
    for input_seq, target_seq in tqdm(
        val_loader, total=len(val_loader), desc='Validation: ', leave=False
    ):
        # Move hidden state, input and output sequences to device
        hidden = model.init_hidden(1).to(device)
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)

        output_seq, hidden = model(input_seq, hidden)

        loss_value = loss(output_seq, target_seq)
        val_loss += loss_value.item()

    val_loss /= len(val_loader)
    return val_loss


def plot_piano_roll(notes: pd.DataFrame, time: Optional[float] = None):
    if time is not None:
        title = f'First {time} seconds'
    else:
        time = notes['end'].max()
        title = 'All notes'
    
    notes_cropped = notes[notes['end'] <= time]

    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes_cropped['pitch'], notes_cropped['pitch']], axis=0)
    plot_start_stop = np.stack([notes_cropped['start'], notes_cropped['end']], axis=0)

    plt.plot(
        plot_start_stop, plot_pitch, color='b', marker='.'
    )
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    plt.title(title)


if __name__ == '__main__':
    # Load config
    with open(TRAIN_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    device = config['device']

    # Create tensorboard writer
    writer = SummaryWriter(f'runs/{config["experiment_name"]}')    

    # Create datasets and loaders
    train_dataset = NotesDataset(pd.read_csv(config['train_tokens_path']))
    val_dataset = NotesDataset(pd.read_csv(config['val_tokens_path']))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    # Create model
    model = RNN(PITCH_CNT, config['model']['hidden_size'], config['model']['num_layers']).to(device)

    # Load pretrained weights if specified
    if config['pretrained_weights_path'] is not None:
        model.load_state_dict(torch.load(config['pretrained_weights_path']))

    # Create loss, optimizer and scheduler
    loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.ones((128,)) * config['loss']['pos_weight']
    ).type(torch.cuda.FloatTensor)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer']['weight_decay'],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        config['scheduler']['period'],
        config['scheduler']['factor'],
    )

    # Select first validation file as sample for generation and get it's input tensor
    sample_tensor = val_dataset[0][0]
    sample_tensor_start = val_dataset[0][0][:10]

    best_val_loss = None
    for epoch in range(config['num_epochs']):
        print(f'\nEpoch {epoch + 1}')

        # Train model for one epoch
        train_loss = train_epoch(
            model,
            loss,
            optimizer,
            train_loader,
            device,
            scheduler,
            epoch,
            writer,
        )
        print(f'Train loss: {train_loss}')

        # Calculate validation loss
        val_loss = validate_model(model, loss, val_loader, device)
        print(f'Validation loss: {val_loss}')

        # Write validation loss to tensorobard
        writer.add_scalar(
            'validation loss',
            val_loss,
            epoch,
        )

        # Add sample and generated piano roll to tensorboard
        sample_notes = generate_notes(
            model,
            sample_tensor,
            15,
            config['resolution'],
            device,
        )
        plot_piano_roll(sample_notes, time=15)
        writer.add_figure('Sample piano roll', plt.gcf(), global_step=epoch)

        generated_notes = generate_notes(
            model,
            sample_tensor_start,
            15,
            config['resolution'],
            device,
        )
        plot_piano_roll(generated_notes, time=15)
        writer.add_figure('Generated piano roll', plt.gcf(), global_step=epoch)

        # Save best model
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'results/models/{config["experiment_name"]}.pt')
