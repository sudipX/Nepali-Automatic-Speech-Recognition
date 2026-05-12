"""
Tokenizer for Nepali ASR
Supports both SentencePiece and character-level tokenization
"""

import sentencepiece as spm
import json
import unicodedata
import re
from pathlib import Path


class NepaliTokenizer:
    """
    Tokenizer for Nepali text.
    Supports SentencePiece and character-level tokenization.
    """
    def __init__(self, model_path):
        """
        Args:
            model_path: Path to tokenizer model (.model for SentencePiece, .json for character)
        """
        self.model_path = Path(model_path)
        
        if self.model_path.suffix == '.model':
            # SentencePiece tokenizer
            self.tokenizer_type = 'sentencepiece'
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(str(model_path))
            self.vocab_size = self.sp.vocab_size()
        elif self.model_path.suffix == '.json':
            # Character tokenizer
            self.tokenizer_type = 'character'
            with open(model_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.vocab = data['vocab']
            self.char2idx = data['char2idx']
            self.idx2char = {int(k): v for k, v in data['idx2char'].items()}
            self.vocab_size = len(self.vocab)
        else:
            raise ValueError(f"Unsupported tokenizer format: {self.model_path.suffix}")
    
    def encode(self, text):
        """
        Encode text to token IDs.
        
        Args:
            text: Nepali text string
        Returns:
            List of token IDs
        """
        # Normalize text
        text = self.normalize_text(text)
        
        if self.tokenizer_type == 'sentencepiece':
            return self.sp.encode(text, out_type=int)
        else:
            return [self.char2idx.get(c, 1) for c in text]  # 1 = <unk>
    
    def decode(self, tokens):
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs
        Returns:
            Decoded text string
        """
        if self.tokenizer_type == 'sentencepiece':
            return self.sp.decode(tokens)
        else:
            # Skip special tokens (0=pad, 1=unk, 2=sos, 3=eos)
            return ''.join([self.idx2char.get(t, '') for t in tokens 
                          if t not in [0, 1, 2, 3]])
    
    @staticmethod
    def normalize_text(text):
        """
        Normalize Nepali text.
        
        Args:
            text: Nepali text string
        Returns:
            Normalized text
        """
        # Unicode normalization (CRITICAL for Devanagari!)
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize spaces around Devanagari punctuation
        text = re.sub(r'\s*([।॥])\s*', r'\1 ', text)
        
        return text.strip()


def train_sentencepiece_tokenizer(
    transcripts,
    output_prefix='nepali_tokenizer',
    vocab_size=3000,
    character_coverage=1.0
):
    """
    Train SentencePiece tokenizer for Nepali.
    
    Args:
        transcripts: List of Nepali text transcripts
        output_prefix: Output model prefix
        vocab_size: Vocabulary size (2000-5000 recommended for Nepali)
        character_coverage: Must be 1.0 for Devanagari!
    Returns:
        Path to trained model
    """
    # Normalize all transcripts
    normalized_transcripts = []
    for text in transcripts:
        normalized = NepaliTokenizer.normalize_text(text)
        if normalized:
            normalized_transcripts.append(normalized)
    
    # Write to temporary file
    temp_file = 'temp_nepali_text.txt'
    with open(temp_file, 'w', encoding='utf-8') as f:
        for text in normalized_transcripts:
            f.write(text + '\n')
    
    # Train tokenizer
    training_args = [
        f'--input={temp_file}',
        f'--model_prefix={output_prefix}',
        f'--vocab_size={vocab_size}',
        '--model_type=unigram',
        f'--character_coverage={character_coverage}',
        '--pad_id=0',
        '--unk_id=1',
        '--bos_id=2',
        '--eos_id=3',
        '--normalization_rule_name=nmt_nfkc_cf',
        '--user_defined_symbols=।,॥',  # Devanagari punctuation
        '--max_sentence_length=4192',
        '--shuffle_input_sentence=true',
    ]
    
    spm.SentencePieceTrainer.train(' '.join(training_args))
    
    # Clean up
    Path(temp_file).unlink()
    
    print(f"✓ Tokenizer trained: {output_prefix}.model")
    return output_prefix + '.model'


def train_character_tokenizer(transcripts, output_path='char_tokenizer.json'):
    """
    Train character-level tokenizer for Nepali.
    
    Args:
        transcripts: List of Nepali text transcripts
        output_path: Output path for tokenizer JSON
    Returns:
        Path to trained tokenizer
    """
    # Collect all unique characters
    all_chars = set()
    for text in transcripts:
        normalized = NepaliTokenizer.normalize_text(text)
        all_chars.update(normalized)
    
    # Create vocabulary
    vocab = ['<pad>', '<unk>', '<sos>', '<eos>']
    vocab.extend(sorted(all_chars))
    
    # Create mappings
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for i, c in enumerate(vocab)}
    
    # Save
    tokenizer_data = {
        'type': 'character',
        'vocab': vocab,
        'char2idx': char2idx,
        'idx2char': {int(k): v for k, v in idx2char.items()}
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Character tokenizer trained: {output_path}")
    print(f"  Vocabulary size: {len(vocab)}")
    
    return output_path