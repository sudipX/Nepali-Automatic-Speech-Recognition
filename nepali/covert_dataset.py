"""
Convert YOUR Nepali Dataset
CSV: E:\DATASET\metadata_final.csv
Audio: E:\DATASET\nepali_merged
"""

import pandas as pd
import os
from pathlib import Path
import shutil

def convert_your_dataset():
    """
    Convert your specific Nepali dataset.
    """
    print("="*80)
    print("CONVERTING YOUR NEPALI DATASET")
    print("="*80)
    
    # Your specific paths
    csv_file = r"E:\DATASET\metadata_final.csv"
    audio_base = r"E:\DATASET\nepali_merged"
    output_dir = Path(".")  # Current directory (C:\Users\AJIT\Conformer\nepali)
    
    # Step 1: Check CSV exists
    print("\n[Step 1] Checking CSV file...")
    if not Path(csv_file).exists():
        print(f"❌ CSV not found: {csv_file}")
        print("\nPlease check:")
        print(f"  - Is the path correct?")
        print(f"  - Is it 'metadata_final.csv' or 'metadat_final.csv'?")
        print(f"\nTry: dir E:\\DATASET\\*.csv")
        return False
    
    print(f"✓ Found: {csv_file}")
    
    # Step 2: Read CSV
    print(f"\n[Step 2] Reading CSV...")
    try:
        # Try different encodings
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(csv_file, encoding='utf-16')
            except:
                df = pd.read_csv(csv_file, encoding='latin-1')
        
        print(f"✓ Loaded {len(df)} rows")
        print(f"\nColumns found: {list(df.columns)}")
        print(f"\nFirst 3 rows:")
        print(df.head(3).to_string())
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False
    
    # Step 3: Auto-detect or ask for columns
    print(f"\n[Step 3] Detecting column names...")
    
    # Common column name variations
    audio_col = None
    text_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        # Audio column
        if any(x in col_lower for x in ['audio', 'file', 'path', 'wav', 'filename', 'name']):
            audio_col = col
            print(f"  Found audio column: '{col}'")
        # Text column  
        if any(x in col_lower for x in ['text', 'transcript', 'sentence', 'nepali', 'label']):
            text_col = col
            print(f"  Found text column: '{col}'")
    
    # If not auto-detected, show options
    if audio_col is None or text_col is None:
        print("\n⚠️  Could not auto-detect columns. Please choose:")
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
            # Show sample value
            sample = str(df[col].iloc[0])[:50]
            print(f"     Sample: {sample}")
            print()
        
        print("Please edit this script and set:")
        print(f"  audio_col = '{df.columns[0]}'  # Change this")
        print(f"  text_col = '{df.columns[1]}'   # Change this")
        return False
    
    print(f"\n✓ Will use:")
    print(f"  Audio column: '{audio_col}'")
    print(f"  Text column:  '{text_col}'")
    
    # Step 4: Find audio files
    print(f"\n[Step 4] Looking for audio files...")
    
    # Check different possible locations
    possible_dirs = [
        Path(audio_base),
        Path(audio_base) / "audio",
        Path(audio_base) / "wav",
        Path(audio_base) / "clips",
        Path(r"E:\DATASET"),
        Path(r"E:\DATASET\audio"),
    ]
    
    audio_dir = None
    sample_filename = df[audio_col].iloc[0]
    
    print(f"  Looking for: {sample_filename}")
    
    for test_dir in possible_dirs:
        if test_dir.exists():
            test_file = test_dir / sample_filename
            if test_file.exists():
                audio_dir = test_dir
                print(f"  ✓ Found in: {test_dir}")
                break
            else:
                print(f"  ✗ Not in: {test_dir}")
    
    if audio_dir is None:
        print("\n❌ Could not find audio files!")
        print(f"\nSearched in:")
        for d in possible_dirs:
            print(f"  - {d}")
        print(f"\nPlease check where your .wav files are located:")
        print(f"  dir E:\\DATASET\\nepali_merged\\*.wav")
        return False
    
    # Count existing files
    print(f"\n[Step 5] Counting audio files...")
    audio_files_exist = 0
    missing_files = []
    
    for idx, filename in enumerate(df[audio_col]):
        if (audio_dir / filename).exists():
            audio_files_exist += 1
        else:
            if len(missing_files) < 5:
                missing_files.append(filename)
        
        if (idx + 1) % 5000 == 0:
            print(f"  Checked {idx + 1}/{len(df)} files...")
    
    print(f"\n✓ Found {audio_files_exist}/{len(df)} audio files")
    
    if missing_files:
        print(f"\nSome missing files (showing first 5):")
        for f in missing_files:
            print(f"  - {f}")
    
    if audio_files_exist == 0:
        print("\n❌ No audio files found at all!")
        return False
    
    # Step 6: Create output structure
    print(f"\n[Step 6] Creating output structure...")
    
    raw_data_dir = output_dir / "raw_data"
    raw_audio_dir = raw_data_dir / "audio"
    raw_audio_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created: {raw_data_dir}")
    
    # Step 7: Create transcripts file (NO COPYING - just reference)
    print(f"\n[Step 7] Creating transcripts file...")
    print("  (Referencing audio files in place - NOT copying)")
    
    transcript_lines = []
    valid_count = 0
    
    for idx, row in df.iterrows():
        audio_filename = row[audio_col]
        transcript = str(row[text_col]).strip()
        
        # Check if audio file exists
        src_audio = audio_dir / audio_filename
        
        if not src_audio.exists():
            continue
        
        # Reference the original location (don't copy)
        # We'll create symlinks or update manifests to use original paths
        transcript_lines.append(f"{src_audio}|{transcript}\n")
        valid_count += 1
        
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} rows...")
    
    print(f"✓ Created {valid_count} transcript entries")
    
    # Write transcripts file
    transcripts_file = raw_data_dir / "transcripts.txt"
    with open(transcripts_file, 'w', encoding='utf-8') as f:
        f.writelines(transcript_lines)
    
    print(f"✓ Saved: {transcripts_file}")
    
    # Step 8: Show summary
    print("\n" + "="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print(f"\nDataset statistics:")
    print(f"  ✓ Total samples: {valid_count}")
    print(f"  ✓ Audio location: {audio_dir}")
    print(f"  ✓ Transcripts file: {transcripts_file}")
    
    print(f"\nSample (first 3):")
    for line in transcript_lines[:3]:
        parts = line.strip().split('|')
        if len(parts) == 2:
            print(f"  Audio: {Path(parts[0]).name}")
            print(f"  Text:  {parts[1][:60]}...")
            print()
    
    print("="*80)
    print("NEXT STEPS:")
    print("="*80)
    
    print("\n1. Prepare data (train tokenizer + create manifests):")
    print(f"   python prepare_data.py ^")
    print(f"       --task tokenizer ^")
    print(f"       --transcripts {transcripts_file}")
    
    print("\n   Then create manifests:")
    print(f"   python create_manifests_from_full_paths.py")
    
    print("\n2. Start training:")
    print(f"   python train.py --config configs\\config.yaml")
    
    print("\n" + "="*80)
    
    return True


if __name__ == '__main__':
    convert_your_dataset()