"""
Create Manifest Files from Full Audio Paths
"""

import json
from pathlib import Path
import random

def create_manifests_from_full_paths(transcript_file, output_dir, split_ratio=(0.9, 0.05, 0.05)):
    """Create train/val/test manifests from transcripts file with full paths."""
    print("="*80)
    print("CREATING MANIFEST FILES")
    print("="*80)
    
    # Read transcripts
    print(f"\nReading: {transcript_file}")
    data = []
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if '|' not in line:
                continue
            
            parts = line.strip().split('|', 1)
            if len(parts) != 2:
                continue
            
            audio_path, transcript = parts
            
            # Verify audio exists
            if not Path(audio_path).exists():
                if line_num % 10000 == 0:
                    print(f"  Warning: Audio not found (line {line_num}): {audio_path}")
                continue
            
            data.append({
                'audio_path': audio_path,
                'transcript': transcript.strip()
            })
            
            if line_num % 10000 == 0:
                print(f"  Processed {line_num} lines, {len(data)} valid...")
    
    print(f"\n✓ Loaded {len(data)} valid samples")
    
    if len(data) == 0:
        print("❌ No valid data found!")
        return False
    
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    # Split
    train_ratio, val_ratio, test_ratio = split_ratio
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_data)} samples ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_data)} samples ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_data)} samples ({test_ratio*100:.0f}%)")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save manifests
    manifests = {
        'train': (output_dir / 'train_manifest.json', train_data),
        'val': (output_dir / 'val_manifest.json', val_data),
        'test': (output_dir / 'test_manifest.json', test_data)
    }
    
    for split_name, (manifest_path, split_data) in manifests.items():
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Saved: {manifest_path}")
    
    # Show samples
    print(f"\nSample from train set:")
    for i in range(min(3, len(train_data))):
        sample = train_data[i]
        print(f"\n  [{i+1}]")
        print(f"  Audio: {Path(sample['audio_path']).name}")
        print(f"  Text:  {sample['transcript'][:60]}...")
    
    print("\n" + "="*80)
    print("MANIFESTS CREATED SUCCESSFULLY!")
    print("="*80)
    
    return True


if __name__ == '__main__':
    transcript_file = 'raw_data/transcripts.txt'
    output_dir = 'data'
    
    if not Path(transcript_file).exists():
        print(f"❌ Transcript file not found: {transcript_file}")
        print("\nRun convert_your_dataset.py first!")
    else:
        create_manifests_from_full_paths(transcript_file, output_dir)