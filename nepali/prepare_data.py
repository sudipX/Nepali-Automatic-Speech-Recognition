"""
Prepare Data for Nepali Conformer ASR
1. Train tokenizer
2. Create manifest files
"""

import argparse
from pathlib import Path
from data import train_sentencepiece_tokenizer, train_character_tokenizer, create_manifest


def prepare_tokenizer(transcript_file, output_prefix, tokenizer_type='sentencepiece', vocab_size=3000):
    """
    Train tokenizer on transcripts.
    
    Args:
        transcript_file: Path to file with transcripts (one per line)
        output_prefix: Output model prefix
        tokenizer_type: 'sentencepiece' or 'character'
        vocab_size: Vocabulary size (for sentencepiece)
    """
    print("="*80)
    print("TRAINING TOKENIZER")
    print("="*80)
    
    # Load transcripts
    print(f"\nLoading transcripts from: {transcript_file}")
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcripts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(transcripts)} transcripts")
    print(f"Sample: {transcripts[0]}")
    
    # Train tokenizer
    if tokenizer_type == 'sentencepiece':
        print(f"\nTraining SentencePiece tokenizer...")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Output: {output_prefix}.model")
        
        model_path = train_sentencepiece_tokenizer(
            transcripts,
            output_prefix=output_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0  # CRITICAL for Devanagari!
        )
        
        print(f"\n✓ Tokenizer saved to: {model_path}")
        
    elif tokenizer_type == 'character':
        print(f"\nTraining character tokenizer...")
        print(f"  Output: {output_prefix}.json")
        
        model_path = train_character_tokenizer(
            transcripts,
            output_path=output_prefix + '.json'
        )
        
        print(f"\n✓ Tokenizer saved to: {model_path}")
    
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    # Test tokenizer
    print(f"\nTesting tokenizer...")
    from data import NepaliTokenizer
    tokenizer = NepaliTokenizer(model_path)
    
    test_text = transcripts[0]
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"  Original:  {test_text}")
    print(f"  Encoded:   {encoded[:20]}... (showing first 20 tokens)")
    print(f"  Decoded:   {decoded}")
    print(f"  Match:     {'✓' if test_text == decoded else '❌'}")
    
    if test_text != decoded:
        print("\n  ⚠️  WARNING: Decoded text doesn't match original!")
        print("  This may indicate a tokenizer configuration issue.")
    
    print("\n" + "="*80)
    return model_path


def prepare_manifest(audio_dir, transcript_file, output_dir, split_ratio=(0.9, 0.05, 0.05)):
    """
    Create train/val/test manifest files.
    
    Args:
        audio_dir: Directory containing audio files
        transcript_file: File with format: filename|transcript
        output_dir: Output directory for manifests
        split_ratio: Tuple of (train, val, test) ratios
    """
    print("="*80)
    print("CREATING MANIFEST FILES")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {transcript_file}")
    audio_dir = Path(audio_dir)
    data = []
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' not in line:
                continue
            
            filename, transcript = line.strip().split('|', 1)
            audio_path = audio_dir / filename
            
            if not audio_path.exists():
                print(f"  Warning: Audio file not found: {audio_path}")
                continue
            
            data.append({
                'audio_path': str(audio_path),
                'transcript': transcript.strip()
            })
    
    print(f"Loaded {len(data)} samples")
    
    # Split data
    import random
    random.shuffle(data)
    
    train_ratio, val_ratio, test_ratio = split_ratio
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"\nSplitting data:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")
    
    # Save manifests
    import json
    
    manifests = {
        'train': (output_dir / 'train_manifest.json', train_data),
        'val': (output_dir / 'val_manifest.json', val_data),
        'test': (output_dir / 'test_manifest.json', test_data)
    }
    
    for split_name, (manifest_path, split_data) in manifests.items():
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {split_name}: {manifest_path}")
    
    print("\n" + "="*80)
    return manifests


def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description='Prepare data for Nepali ASR')
    parser.add_argument('--task', type=str, required=True,
                       choices=['tokenizer', 'manifest', 'all'],
                       help='Task to perform')
    
    # Tokenizer arguments
    parser.add_argument('--transcripts', type=str,
                       help='Path to transcript file (one per line)')
    parser.add_argument('--tokenizer-output', type=str, default='checkpoints/nepali_tokenizer',
                       help='Output prefix for tokenizer')
    parser.add_argument('--tokenizer-type', type=str, default='sentencepiece',
                       choices=['sentencepiece', 'character'],
                       help='Type of tokenizer')
    parser.add_argument('--vocab-size', type=int, default=3000,
                       help='Vocabulary size for sentencepiece')
    
    # Manifest arguments
    parser.add_argument('--audio-dir', type=str,
                       help='Directory containing audio files')
    parser.add_argument('--transcript-file', type=str,
                       help='File with format: filename|transcript')
    parser.add_argument('--manifest-output', type=str, default='data',
                       help='Output directory for manifest files')
    
    args = parser.parse_args()
    
    # Prepare tokenizer
    if args.task in ['tokenizer', 'all']:
        if not args.transcripts:
            parser.error('--transcripts required for tokenizer task')
        
        prepare_tokenizer(
            args.transcripts,
            args.tokenizer_output,
            args.tokenizer_type,
            args.vocab_size
        )
    
    # Prepare manifests
    if args.task in ['manifest', 'all']:
        if not args.audio_dir or not args.transcript_file:
            parser.error('--audio-dir and --transcript-file required for manifest task')
        
        prepare_manifest(
            args.audio_dir,
            args.transcript_file,
            args.manifest_output
        )
    
    print("\n✓ Data preparation complete!")


if __name__ == '__main__':
    main()