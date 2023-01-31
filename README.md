# Music Generation with RNN

### Setup

1. Install fluidsynth on your system
2. Install requirements: ```pip install -r requirements.txt```

### Training models

1. Download dataset: ```python3 -m src.download_dataset```
2. Modify config that is located in ```config/data_preprocessing.yaml```
3. Run data preprocessing script: ```python3 -m src.data_preprocessing```
4. Modify config that is located in ```config/train.yaml```
5. Start training: ```python3 -m src.train```

**Steps 2-3 can be done only ones.**

Optionally you can launch tensorboard to track learning curves and generated piano rolls.
Logging folder is ```runs``` in the project root.

### Generating audio

1. Modify config that is located in ```config/generate.yaml```
2. Run generation script: ```python3 -m src.generate```

The script generates only MIDI files that should be converted to MP3 or WAV.
Additionaly it creates piano rolls for sample file notes and generated notes.
