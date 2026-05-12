"""
Training Script for Nepali Conformer ASR
Optimized for 50k Nepali Speech dataset to achieve 15-18% CER
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import argparse

from model import build_model
from data import NepaliASRDataset, collate_fn
from utils import MetricsTracker, create_scheduler


def validate(model, val_loader, tokenizer, device, max_samples=None):
    """
    Validate model on validation set.
    
    Args:
        model: Conformer model
        val_loader: Validation DataLoader
        tokenizer: Tokenizer for decoding
        device: Device to run on
        max_samples: Maximum samples to validate (None = all)
    Returns:
        Dictionary with metrics
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validating')):
            if max_samples and batch_idx * val_loader.batch_size >= max_samples:
                break
            
            mel_specs, mel_lengths, targets, target_lengths = batch
            mel_specs = mel_specs.to(device)
            mel_lengths = mel_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward pass
            log_probs, output_lengths = model(mel_specs, mel_lengths)
            loss = model.compute_loss(log_probs, targets, output_lengths, target_lengths)
            
            # Decode predictions
            preds = model.decode(log_probs, output_lengths)
            
            # Convert to text
            pred_texts = [tokenizer.decode(pred) for pred in preds]
            ref_texts = [tokenizer.decode(targets[i][:target_lengths[i]].cpu().tolist()) 
                        for i in range(len(targets))]
            
            # Update metrics
            metrics_tracker.update(pred_texts, ref_texts, loss.item())
    
    # Compute final metrics
    metrics = metrics_tracker.compute()
    
    # Print samples
    print(f"\n{'='*80}")
    print(f"Validation Results:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  CER: {metrics['cer']:.2f}%")
    print(f"  WER: {metrics['wer']:.2f}%")
    print(f"\nSample Predictions:")
    for i in range(min(3, len(pred_texts))):
        print(f"\n  [{i+1}]")
        print(f"  REF: {ref_texts[i]}")
        print(f"  HYP: {pred_texts[i]}")
    print(f"{'='*80}\n")
    
    model.train()
    return metrics


def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, config):
    """
    Train for one epoch.
    
    Args:
        model: Conformer model
        train_loader: Training DataLoader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: GradScaler for mixed precision
        device: Device to run on
        epoch: Current epoch number
        config: Configuration dictionary
    Returns:
        Average loss for epoch
    """
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    max_grad_norm = config['training'].get('max_grad_norm', 5.0)
    use_amp = config['training'].get('use_amp', True) and device.type == 'cuda'
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        mel_specs, mel_lengths, targets, target_lengths = batch
        mel_specs = mel_specs.to(device)
        mel_lengths = mel_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        # Forward pass
        if use_amp:
            with autocast():
                log_probs, output_lengths = model(mel_specs, mel_lengths)
                loss = model.compute_loss(log_probs, targets, output_lengths, target_lengths)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
        else:
            log_probs, output_lengths = model(mel_specs, mel_lengths)
            loss = model.compute_loss(log_probs, targets, output_lengths, target_lengths)
            loss = loss / accumulation_steps
            loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'lr': f'{current_lr:.6f}'
        })
    
    return epoch_loss / num_batches


def train(config_path):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device setup
    device = torch.device(config['system']['device'])
    print(f"Using device: {device}")
    
    # Set seed
    if 'seed' in config['system']:
        torch.manual_seed(config['system']['seed'])
    
    # Build model
    print("\nBuilding model...")
    model = build_model(config).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {num_params:.2f}M")
    
    # Load tokenizer (from dataset)
    from data import NepaliTokenizer
    tokenizer = NepaliTokenizer(config['tokenizer']['model_path'])
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = NepaliASRDataset(
        config['data']['train_manifest'],
        config,
        augment=True
    )
    val_dataset = NepaliASRDataset(
        config['data']['val_manifest'],
        config,
        augment=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['system'].get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['system'].get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=tuple(config['training'].get('betas', [0.9, 0.98])),
        eps=config['training'].get('eps', 1e-9),
        weight_decay=config['training'].get('weight_decay', 1e-6)
    )
    
    # Learning rate scheduler
    scheduler = create_scheduler(optimizer, config)
    print(f"Scheduler: {type(scheduler).__name__}")
    
    # Mixed precision
    use_amp = config['training'].get('use_amp', True) and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    print(f"Mixed precision training: {use_amp}")
    
    # Training setup
    num_epochs = config['training']['num_epochs']
    patience = config['training'].get('patience', 20)
    save_dir = Path(config['checkpoint']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_cer = float('inf')
    patience_counter = 0
    
    if config['checkpoint'].get('resume_from'):
        print(f"\nResuming from: {config['checkpoint']['resume_from']}")
        checkpoint = torch.load(config['checkpoint']['resume_from'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_cer = checkpoint.get('cer', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best CER: {best_cer:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"{'='*80}\n")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        avg_train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, 
            scaler, device, epoch + 1, config
        )
        
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validate every N epochs
        val_every = config['training'].get('val_every_n_epochs', 5)
        if (epoch + 1) % val_every == 0 or epoch == num_epochs - 1:
            val_metrics = validate(model, val_loader, tokenizer, device)
            
            # Save best model
            if val_metrics['cer'] < best_cer:
                best_cer = val_metrics['cer']
                patience_counter = 0
                
                best_model_path = save_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'cer': val_metrics['cer'],
                    'wer': val_metrics['wer'],
                    'config': config
                }, best_model_path)
                
                print(f"✓ New best model saved! CER: {val_metrics['cer']:.2f}%")
            else:
                patience_counter += val_every
                print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                break
        
        # Save checkpoint every N epochs
        save_every = config['checkpoint'].get('save_every', 10)
        if (epoch + 1) % save_every == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best CER: {best_cer:.2f}%")
    print(f"Best model saved at: {save_dir / 'best_model.pt'}")
    print(f"{'='*80}\n")


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description='Train Nepali Conformer ASR')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    args = parser.parse_args()
    
    train(args.config)


if __name__ == '__main__':
    main()