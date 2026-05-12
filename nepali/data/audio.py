"""
Audio Processing for Nepali ASR
"""

import torch
import torchaudio
import torchaudio.transforms as T


class AudioTransform:
    """
    Audio feature extraction and preprocessing.
    """
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary with audio settings
        """
        audio_config = config['audio']
        
        self.sample_rate = audio_config['sample_rate']
        self.n_mels = audio_config['n_mels']
        self.n_fft = audio_config['n_fft']
        self.hop_length = audio_config['hop_length']
        self.win_length = audio_config['win_length']
        self.fmin = audio_config.get('fmin', 0)
        self.fmax = audio_config.get('fmax', 8000)
        
        # Mel spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.fmin,
            f_max=self.fmax,
            n_mels=self.n_mels,
            power=2.0
        )
    
    def __call__(self, waveform):
        """
        Convert waveform to mel spectrogram.
        
        Args:
            waveform: (channels, time) or (time,)
        Returns:
            mel_spec: (time, n_mels)
        """
        # Ensure 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Compute mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Transpose to (time, n_mels)
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)
        
        return mel_spec


def load_audio(audio_path, sample_rate=16000):
    """
    Load audio file and resample if necessary.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
    Returns:
        waveform: (channels, time)
    """
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    return waveform


def compute_mel_spectrogram(waveform, config):
    """
    Compute mel spectrogram from waveform.
    
    Args:
        waveform: (channels, time) or (time,)
        config: Configuration dictionary
    Returns:
        mel_spec: (time, n_mels)
    """
    audio_transform = AudioTransform(config)
    return audio_transform(waveform)