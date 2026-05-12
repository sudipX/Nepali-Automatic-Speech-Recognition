"""
Dataset for Nepali ASR
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from .audio import load_audio, AudioTransform
from .tokenizer import NepaliTokenizer
from .data_augmentation import build_augmentation


class NepaliASRDataset(Dataset):
    """
    Dataset for Nepali Automatic Speech Recognition.
    
    Expects a manifest file in JSON format with entries like:
    {
        "audio_path": "path/to/audio.wav",
        "transcript": "नेपाली transcript text",
        "duration": 5.2  (optional)
    }
    """
    def __init__(self, manifest_path, config, augment=True):
        """
        Args:
            manifest_path: Path to manifest JSON file
            config: Configuration dictionary
            augment: Whether to apply data augmentation
        """
        self.config = config
        self.augment = augment
        
        # Load manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Initialize audio transform
        self.audio_transform = AudioTransform(config)
        
        # Initialize tokenizer
        tokenizer_path = config['tokenizer']['model_path']
        self.tokenizer = NepaliTokenizer(tokenizer_path)
        
        # Initialize augmentation
        if augment:
            self.augmentation = build_augmentation(config)
        else:
            self.augmentation = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            mel_spec: (time, n_mels)
            mel_length: int
            target: (target_length,)
            target_length: int
        """
        item = self.data[idx]
        
        # Load audio
        audio_path = item['audio_path']
        waveform = load_audio(audio_path, self.audio_transform.sample_rate)
        
        # Convert to mel spectrogram
        mel_spec = self.audio_transform(waveform)
        
        # Apply augmentation if enabled
        if self.augmentation is not None:
            mel_spec = self.augmentation(mel_spec)
        
        # Get transcript
        transcript = item['transcript']
        
        # Tokenize
        tokens = self.tokenizer.encode(transcript)
        target = torch.tensor(tokens, dtype=torch.long)
        
        # Get lengths
        mel_length = mel_spec.shape[0]
        target_length = len(tokens)
        
        return mel_spec, mel_length, target, target_length


def collate_fn(batch):
    """
    Collate function for DataLoader.
    Pads sequences to the same length within a batch.
    
    Args:
        batch: List of tuples (mel_spec, mel_length, target, target_length)
    Returns:
        mel_specs: (batch, max_mel_length, n_mels)
        mel_lengths: (batch,)
        targets: (batch, max_target_length)
        target_lengths: (batch,)
    """
    # Sort by mel_length (descending) for efficient packing
    batch = sorted(batch, key=lambda x: x[1], reverse=True)
    
    mel_specs, mel_lengths, targets, target_lengths = zip(*batch)
    
    # Get dimensions
    batch_size = len(mel_specs)
    max_mel_length = max(mel_lengths)
    n_mels = mel_specs[0].shape[1]
    max_target_length = max(target_lengths)
    
    # Pad mel spectrograms
    padded_mel_specs = torch.zeros(batch_size, max_mel_length, n_mels)
    for i, mel_spec in enumerate(mel_specs):
        length = mel_spec.shape[0]
        padded_mel_specs[i, :length, :] = mel_spec
    
    # Pad targets
    padded_targets = torch.zeros(batch_size, max_target_length, dtype=torch.long)
    for i, target in enumerate(targets):
        length = len(target)
        padded_targets[i, :length] = target
    
    # Convert lengths to tensors
    mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return padded_mel_specs, mel_lengths, padded_targets, target_lengths


def create_manifest(audio_dir, transcript_file, output_manifest):
    """
    Create manifest file from audio directory and transcript file.
    
    Args:
        audio_dir: Directory containing audio files
        transcript_file: Text file with format: filename|transcript
        output_manifest: Output path for manifest JSON
    """
    audio_dir = Path(audio_dir)
    data = []
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' not in line:
                continue
            
            filename, transcript = line.strip().split('|', 1)
            audio_path = audio_dir / filename
            
            if not audio_path.exists():
                print(f"Warning: Audio file not found: {audio_path}")
                continue
            
            data.append({
                'audio_path': str(audio_path),
                'transcript': transcript.strip()
            })
    
    # Save manifest
    with open(output_manifest, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Created manifest with {len(data)} samples: {output_manifest}")
    return output_manifest