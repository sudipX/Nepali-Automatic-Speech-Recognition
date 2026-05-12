# Nepali Conformer ASR

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A Deep Learning based Automatic Speech Recognition (ASR) system for Nepali language using the Conformer architecture and Connectionist Temporal Classification (CTC).

</div>

---

## Overview

This project implements a complete end-to-end Nepali Speech Recognition pipeline using the Conformer architecture, combining the strengths of Convolutional Neural Networks and Transformers for robust speech understanding.

The system includes:

- Conformer-based ASR architecture
- Nepali tokenizer using SentencePiece
- Training and inference pipeline
- Audio preprocessing and augmentation
- Streamlit based web interface
- Dataset preparation utilities
- Mixed precision training support

The model is trained on Nepali speech data and optimized for accurate transcription performance.

---

## Model Architecture

The architecture is based on the Conformer network, which combines:

- Self-Attention for global context understanding
- Convolution modules for local feature extraction
- Feed Forward layers for representation learning
- CTC decoding for alignment-free speech recognition

### Configuration Highlights

| Component | Value |
|---|---|
| Encoder Dimension | 256 |
| Conformer Blocks | 12 |
| Attention Heads | 4 |
| Mel Features | 80 |
| Sample Rate | 16 kHz |
| Vocabulary Size | 3000 |
| Epochs | 150 |

---

## Project Structure

```bash
Conformer-main/
│
├── nepali/
│   ├── app.py                     # Streamlit web application
│   ├── train.py                   # Training script
│   ├── inference.py               # Inference script
│   ├── prepare_data.py            # Dataset preprocessing
│   ├── create_manifests.py        # Manifest generation
│   ├── config.yaml                # Main configuration
│   │
│   ├── model/
│   │   ├── conformer.py           # Conformer architecture
│   │   ├── model.py               # ASR model wrapper
│   │   └── modules.py             # Supporting modules
│   │
│   ├── data/
│   │   ├── audio.py               # Audio processing
│   │   ├── dataset.py             # Dataset loader
│   │   ├── tokenizer.py           # Tokenization utilities
│   │   └── data_augmentation.py   # SpecAugment implementation
│   │
│   ├── utils/
│   │   ├── metrics.py             # CER/WER metrics
│   │   └── scheduler.py           # Learning rate scheduler
│   │
│   ├── checkpoints/
│   │   └── nepali_tokenizer.model
│   │
│   └── raw_data/
│       └── transcripts.txt
│
└── README.md
```

---

## Demo Video


```html
<iframe
    src="https://drive.google.com/file/d/18CtxePVpVg9nBuYzl06KDUTzWur25sYD/view?usp=sharing"
    width="100%"
    height="480"
    allow="autoplay">
</iframe>
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sudipX/Nepali-Automatic-Speech-Recognition.git
cd your-repository-name
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Linux / macOS

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Prepare your dataset and transcript files.

Expected structure:

```bash
raw_data/
├── audio_1.wav
├── audio_2.wav
└── transcripts.txt
```

Generate manifests:

```bash
python create_manifests.py
```

Prepare processed data:

```bash
python prepare_data.py
```

---

## Training

Run training using:

```bash
python train.py --config config.yaml
```

Features included during training:

- Mixed Precision Training (AMP)
- SpecAugment
- Gradient Accumulation
- Learning Rate Scheduling
- Validation Monitoring
- Checkpoint Saving
- CER/WER Evaluation

---

## Inference

Run transcription on an audio file:

```bash
python inference.py audio.wav \
    --config config.yaml \
    --checkpoint checkpoint_epoch_150.pt
```

---

## Streamlit Web Application

Launch the web interface:

```bash
streamlit run app.py
```

The application supports:

- Audio upload
- Real-time transcription
- Nepali speech recognition
- Interactive UI

---

## Training Techniques Used

### SpecAugment

Data augmentation is used to improve generalization through:

- Frequency masking
- Time masking
- Dynamic augmentation strategies

### SentencePiece Tokenization

A subword tokenizer optimized for Nepali language processing.

### Conformer Encoder

The Conformer architecture improves performance by combining:

- CNN based local feature extraction
- Transformer based long-range modeling

---

## Results

The model is optimized for:

- Low Character Error Rate (CER)
- Robust Nepali transcription
- Efficient inference
- Stable training on medium-scale datasets

You can include sample outputs and evaluation metrics here.

| Metric | Score |
|---|---|
| CER | 9.99% |
| WER | 26.56% |

---

## Technologies Used

- Python
- PyTorch
- Torchaudio
- Streamlit
- SentencePiece
- NumPy
- YAML

---

## Future Improvements

- Real-time streaming ASR
- Transformer language model integration
- Larger Nepali dataset training
- Quantization and deployment optimization
- Mobile deployment
- Multilingual support

---



