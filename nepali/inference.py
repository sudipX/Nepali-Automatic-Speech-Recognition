"""
Inference Script for Nepali Conformer ASR
Transcribe audio files using trained model
"""

import os
import yaml
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

from model import build_model
from data import NepaliTokenizer, load_audio, AudioTransform


def transcribe_audio(audio_path, model, tokenizer, audio_transform, device):
    """
    Transcribe a single audio file.
    
    Args:
        audio_path: Path to audio file
        model: Trained Conformer model
        tokenizer: Tokenizer for decoding
        audio_transform: Audio preprocessing transform
        device: Device to run on
    Returns:
        Transcription text
    """
    # Load audio
    waveform = load_audio(audio_path, audio_transform.sample_rate)
    
    # Convert to mel spectrogram
    mel_spec = audio_transform(waveform)
    
    # Add batch dimension
    mel_spec = mel_spec.unsqueeze(0).to(device)
    mel_length = torch.tensor([mel_spec.size(1)], dtype=torch.long).to(device)
    
    # Inference
    with torch.no_grad():
        log_probs, output_lengths = model(mel_spec, mel_length)
        predictions = model.decode(log_probs, output_lengths)
        transcription = tokenizer.decode(predictions[0])
    
    return transcription


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Transcribe audio with Nepali Conformer ASR')
    parser.add_argument('audio_path', type=str,
                       help='Path to audio file or directory')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save transcriptions (optional)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['system']['device'])
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = NepaliTokenizer(config['tokenizer']['model_path'])
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load model
    print("Loading model...")
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded!")
    
    # Audio transform
    audio_transform = AudioTransform(config)
    
    # Get audio files
    audio_path = Path(args.audio_path)
    if audio_path.is_file():
        audio_files = [audio_path]
    elif audio_path.is_dir():
        audio_files = []
        for ext in ['*.wav', '*.flac', '*.mp3']:
            audio_files.extend(audio_path.glob(f'**/{ext}'))
    else:
        raise ValueError(f"Invalid path: {audio_path}")
    
    print(f"\nTranscribing {len(audio_files)} file(s)...\n")
    
    # Transcribe
    results = []
    for audio_file in tqdm(audio_files):
        try:
            transcription = transcribe_audio(
                str(audio_file), model, tokenizer, audio_transform, device
            )
            print(f"\nFile: {audio_file.name}")
            print(f"Transcription: {transcription}\n")
            results.append((audio_file.name, transcription))
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}\n")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for filename, transcription in results:
                f.write(f"{filename}\t{transcription}\n")
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()