"""
Nepali ASR - Streamlit App
Complete working version for ConformerCTC model
"""

import streamlit as st
import torch
import torchaudio
import numpy as np
import tempfile
import io
from pathlib import Path
import sentencepiece as spm
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import time

# ========== CONFIGURATION ==========
MODEL_PATH = "checkpoint_epoch_150.pt"
TOKENIZER_PATH = "checkpoints/nepali_tokenizer.model"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLE_RATE = 16000
# ===================================


class NepaliASR:
    """ASR inference class for Nepali ConformerCTC"""

    def __init__(self, model_path, tokenizer_path, device=DEVICE):
        self.device = device
        self.sample_rate = SAMPLE_RATE

        # Load tokenizer
        st.info(f"📥 Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        vocab_size = self.tokenizer.vocab_size()
        
        st.success(f"✅ Tokenizer loaded! Vocab size: {vocab_size}")
        st.info(f"📊 Model will use {vocab_size + 1} classes (includes CTC blank token)")

        # Load model checkpoint
        st.info(f"📥 Loading checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            st.info("✅ Found 'model_state_dict' in checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            st.info("✅ Found 'state_dict' in checkpoint")
        else:
            state_dict = checkpoint
            st.info("✅ Using checkpoint directly as state_dict")

        # Import ConformerCTC model
        try:
            from model.model import ConformerCTC
            st.success("✅ Successfully imported ConformerCTC from model.model")
        except ImportError as e:
            st.error(f"❌ Could not import ConformerCTC: {e}")
            st.info("Make sure 'model' directory has __init__.py and model.py")
            raise

        # Create model with CORRECT parameters
        st.info("🔧 Creating ConformerCTC model...")
        self.model = ConformerCTC(
            vocab_size=vocab_size + 1,  # CRITICAL: +1 for CTC blank token!
            input_dim=80,
            encoder_dim=256,
            num_encoder_layers=12,
            num_attention_heads=4,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            dropout=0.15
        )
        st.success(f"✅ Model created with {vocab_size + 1} output classes")

        # Load weights
        st.info("📥 Loading model weights...")
        try:
            self.model.load_state_dict(state_dict, strict=True)
            st.success("✅ Model weights loaded successfully (strict mode)")
        except RuntimeError as e:
            st.warning(f"⚠️ Strict loading failed: {e}")
            st.info("Trying with strict=False (will ignore missing/extra keys)...")
            
            # Load with strict=False and report what's missing/unexpected
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                st.warning(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys:
                st.warning(f"⚠️ Unexpected keys: {unexpected_keys}")
            
            st.success("✅ Model weights loaded (non-strict mode)")
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        st.success(f"✅ Model ready on device: {self.device}")

    def extract_features(self, waveform):
        """Extract log-Mel spectrogram features"""
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=80,
            f_min=0,
            f_max=8000,
        )
        mel = mel_spec(waveform)
        log_mel = torch.log(mel + 1e-10)
        return log_mel.transpose(1, 2)

    def transcribe(self, audio_path=None, audio_array=None):
        """Transcribe audio from file or numpy array"""
        # Load audio
        if audio_path:
            waveform, sr = torchaudio.load(audio_path)
        elif audio_array is not None:
            waveform = torch.from_numpy(audio_array).unsqueeze(0).float()
            sr = self.sample_rate
        else:
            return ""

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        # Extract features
        features = self.extract_features(waveform).to(self.device)

        # Inference - ConformerCTC returns (log_probs, lengths)
        with torch.no_grad():
            log_probs, lengths = self.model(features)

        # Use model's built-in decode method
        decoded_sequences = self.model.decode(log_probs, lengths)
        
        # Get first sequence (batch size is 1)
        if decoded_sequences and len(decoded_sequences) > 0:
            decoded_ids = decoded_sequences[0]
            if decoded_ids:
                return self.tokenizer.DecodeIds(decoded_ids)
        
        return ""


@st.cache_resource
def load_model():
    """Load and cache the ASR model"""
    try:
        model = NepaliASR(MODEL_PATH, TOKENIZER_PATH, DEVICE)
        return model, None
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\n{traceback.format_exc()}"
        return None, error_detail


def record_audio(duration=5, sample_rate=SAMPLE_RATE):
    """Record audio from microphone"""
    audio = sd.rec(
        int(duration * sample_rate), 
        samplerate=sample_rate, 
        channels=1, 
        dtype='float32'
    )
    sd.wait()
    return audio.squeeze()


def save_temp_audio(audio_data, sample_rate=SAMPLE_RATE):
    """Save audio to temp WAV file"""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    if isinstance(audio_data, tuple):
        audio_array, _ = audio_data
    else:
        audio_array = audio_data
    sf.write(tmp_file.name, audio_array, sample_rate)
    return tmp_file.name


def main():
    # Page configuration
    st.set_page_config(
        page_title="Nepali ASR", 
        page_icon="🎤", 
        layout="wide"
    )
    
    # Title
    st.title("🎤 Nepali Speech Recognition")
    st.markdown("### Upload an audio file or record from microphone")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            f"**Nepali ConformerCTC ASR**\n\n"
            f"🔧 Model: ConformerCTC\n\n"
            f"💻 Device: {DEVICE}\n\n"
            f"🏗️ Encoder: 12-layer Conformer\n\n"
            f"📊 Parameters: ~21.68M\n\n"
            f"🎯 Vocab: 3001 (with blank)"
        )
        
        st.header("📊 Settings")
        recording_duration = st.slider(
            "Recording Duration (seconds)", 
            min_value=1, 
            max_value=15, 
            value=5
        )
        
        st.header("📁 File Paths")
        st.code(f"Model: {MODEL_PATH}\nTokenizer: {TOKENIZER_PATH}", language="text")
        
        st.header("🔗 Quick Links")
        st.markdown("- [GitHub](https://github.com)")
        st.markdown("- [Documentation](https://docs.example.com)")

    # Load model
    with st.spinner("🔄 Loading model..."):
        model, error = load_model()
    
    if error:
        st.error("❌ Error loading model!")
        with st.expander("Show error details"):
            st.code(error)
        
        st.info(
            "**Troubleshooting:**\n\n"
            "1. ✅ Check MODEL_PATH exists\n"
            "2. ✅ Check TOKENIZER_PATH exists\n"
            "3. ✅ Verify model/ directory structure\n"
            "4. ✅ Ensure vocab_size matches (3001 with blank)\n"
            "5. ✅ Check imports work: `from model.model import ConformerCTC`"
        )
        return

    # Create tabs
    tab_upload, tab_record = st.tabs(["📁 Upload Audio", "🎤 Record Audio"])

    # ==================== UPLOAD TAB ====================
    with tab_upload:
        st.header("📁 Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file", 
            type=['wav', 'flac', 'mp3', 'ogg', 'm4a'],
            help="Supported formats: WAV, FLAC, MP3, OGG, M4A"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Display audio player
            st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
            
            if st.button("🔮 Transcribe", key="transcribe_upload", type="primary"):
                with st.spinner("🔄 Transcribing..."):
                    try:
                        # Read audio file
                        audio_bytes = uploaded_file.read()
                        uploaded_file.seek(0)
                        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                        
                        # Save to temp file
                        tmp_path = save_temp_audio(audio_data, sr)
                        
                        # Transcribe
                        start_time = time.time()
                        transcription = model.transcribe(audio_path=tmp_path)
                        inference_time = time.time() - start_time
                        
                        # Clean up
                        Path(tmp_path).unlink(missing_ok=True)
                        
                        # Display results
                        st.success("✅ Transcription complete!")
                        
                        st.markdown("### 📝 Transcription Result:")
                        if transcription:
                            st.info(transcription)
                        else:
                            st.warning("⚠️ No speech detected")
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🎵 Audio Duration", f"{len(audio_data)/sr:.2f}s")
                        with col2:
                            st.metric("⏱️ Inference Time", f"{inference_time:.3f}s")
                        with col3:
                            duration = len(audio_data) / sr
                            rtf = inference_time / duration if duration > 0 else 0
                            st.metric("🚀 Real-Time Factor", f"{rtf:.4f}")
                        
                        # Download button
                        if transcription:
                            st.download_button(
                                label="💾 Download Transcription",
                                data=transcription,
                                file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Error during transcription: {str(e)}")
                        with st.expander("Show error details"):
                            import traceback
                            st.code(traceback.format_exc())

    # ==================== RECORD TAB ====================
    with tab_record:
        st.header("🎤 Record from Microphone")
        
        st.info(
            f"**📋 Instructions:**\n\n"
            f"1. Click **'Start Recording'** button below\n"
            f"2. Wait for 3-second countdown\n"
            f"3. Speak clearly in Nepali for **{recording_duration} seconds**\n"
            f"4. Recording stops automatically\n"
            f"5. Click **'Transcribe'** to see the result"
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🔴 Start Recording", key="start_rec", type="primary"):
                with st.spinner(f"🔄 Preparing to record..."):
                    try:
                        # Countdown
                        placeholder = st.empty()
                        for i in range(3, 0, -1):
                            placeholder.warning(f"⏰ Starting in {i}...")
                            time.sleep(1)
                        placeholder.success("🔴 **RECORDING NOW!** Speak in Nepali...")
                        
                        # Record
                        audio = record_audio(recording_duration)
                        st.session_state['recorded_audio'] = audio
                        
                        placeholder.success("✅ Recording complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Recording error: {str(e)}")
                        st.info(
                            "**💡 Troubleshooting:**\n\n"
                            "- Check microphone is connected\n"
                            "- Check browser permissions (click 🔒 in address bar)\n"
                            "- Try refreshing the page\n"
                            "- Close other apps using microphone"
                        )

        # Show recorded audio if exists
        if 'recorded_audio' in st.session_state:
            audio = st.session_state['recorded_audio']
            
            st.success("✅ Audio recorded successfully!")
            
            # Display audio player
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio, SAMPLE_RATE, format='WAV')
            audio_bytes.seek(0)
            st.audio(audio_bytes, format='audio/wav')
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔮 Transcribe", key="transcribe_rec", type="primary"):
                    with st.spinner("🔄 Transcribing..."):
                        try:
                            start_time = time.time()
                            transcription = model.transcribe(audio_array=audio)
                            inference_time = time.time() - start_time
                            
                            # Display results
                            st.success("✅ Transcription complete!")
                            
                            st.markdown("### 📝 Transcription Result:")
                            if transcription:
                                st.info(transcription)
                            else:
                                st.warning("⚠️ No speech detected")
                            
                            # Metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("⏱️ Inference Time", f"{inference_time:.3f}s")
                            with col2:
                                rtf = inference_time / recording_duration
                                st.metric("🚀 Real-Time Factor", f"{rtf:.4f}")
                            
                            # Download buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                audio_bytes_dl = io.BytesIO()
                                sf.write(audio_bytes_dl, audio, SAMPLE_RATE, format='WAV')
                                st.download_button(
                                    label="💾 Download Audio",
                                    data=audio_bytes_dl.getvalue(),
                                    file_name=f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                                    mime="audio/wav"
                                )
                            
                            with col2:
                                if transcription:
                                    st.download_button(
                                        label="💾 Download Text",
                                        data=transcription,
                                        file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                        mime="text/plain"
                                    )
                            
                        except Exception as e:
                            st.error(f"❌ Transcription error: {str(e)}")
                            with st.expander("Show error details"):
                                import traceback
                                st.code(traceback.format_exc())
            
            with col2:
                if st.button("🗑️ Clear Recording", key="clear_rec"):
                    del st.session_state['recorded_audio']
                    st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #888;'>"
        f"Built with ❤️ using Streamlit | Nepali ConformerCTC ASR | "
        f"Device: {DEVICE} | "
        f"<a href='https://github.com' target='_blank'>GitHub</a>"
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()