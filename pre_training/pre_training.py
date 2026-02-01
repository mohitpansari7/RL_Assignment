"""
PART 3: Pretraining - Next-Token Prediction
===========================================

This module implements pretraining for the retail chatbot using
a simple 2-layer transformer model.

GOAL: Learn basic language patterns from retail conversations
METHOD: Next-token prediction (causal language modeling)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import json
from collections import Counter
import matplotlib.pyplot as plt


class SimpleTokenizer:
    """Word-level tokenizer for retail conversations."""
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<BOS>'
        self.EOS_TOKEN = '<EOS>'
        
        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3
    
    def build_vocab(self, texts: List[str]):
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        self.word2idx = {
            self.PAD_TOKEN: self.pad_idx,
            self.UNK_TOKEN: self.unk_idx,
            self.BOS_TOKEN: self.bos_idx,
            self.EOS_TOKEN: self.eos_idx
        }
        
        most_common = word_counts.most_common(self.vocab_size - 4)
        for idx, (word, count) in enumerate(most_common, start=4):
            self.word2idx[word] = idx
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"✓ Vocabulary built: {len(self.word2idx)} tokens")
        print(f"  Most common: {most_common[:10]}")
    
    def encode(self, text: str, max_length: int = 256) -> List[int]:
        words = text.lower().split()
        tokens = [self.bos_idx]
        for word in words:
            tokens.append(self.word2idx.get(word, self.unk_idx))
        tokens.append(self.eos_idx)
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in tokens]
        words = [w for w in words if w not in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]]
        return ' '.join(words)


class ConversationDataset(Dataset):
    """PyTorch Dataset for next-token prediction."""
    
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.sequences = []
        for text in texts:
            tokens = tokenizer.encode(text, max_length)
            if len(tokens) > 1:
                self.sequences.append(tokens)
        
        print(f"✓ Dataset created: {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.sequences[idx]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        return torch.tensor(input_ids, dtype=torch.long), \
               torch.tensor(target_ids, dtype=torch.long)


class SimpleTransformer(nn.Module):
    """Simple 2-layer Transformer for next-token prediction."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, 
                 num_heads: int = 4, num_layers: int = 2, max_len: int = 256):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(max_len, embed_dim) * 0.02
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.positional_encoding[:seq_len, :].unsqueeze(0)
        embeddings = token_embeds + pos_embeds
        
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'), 
            diagonal=1
        ).to(input_ids.device)
        
        hidden_states = self.transformer(embeddings, mask=causal_mask)
        logits = self.output_layer(hidden_states)
        
        return logits
    
    def generate(self, tokenizer: SimpleTokenizer, prompt: str, 
                 max_new_tokens: int = 20, temperature: float = 1.0,
                 device=None) -> str:
        """Generate text with device support."""
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
        
        tokens = tokenizer.encode(prompt, max_length=256)
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
        
        generated_tokens = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                if next_token == tokenizer.eos_idx:
                    break
                
                generated_tokens.append(next_token)
                input_ids = torch.tensor(generated_tokens).unsqueeze(0).to(device)
                
                if len(generated_tokens) >= 256:
                    break
        
        return tokenizer.decode(generated_tokens)


def load_and_prepare_data(csv_path: str, train_ratio: float = 0.8):
    """Load cleaned data from Part 1 and prepare for pretraining."""
    print("=" * 60)
    print("STEP 1: Loading and Preparing Data")
    print("=" * 60)
    
    df = pd.read_csv(csv_path)
    print(f"\n✓ Loaded {len(df)} conversations from {csv_path}")
    
    texts = df['conversation_text'].dropna().tolist()
    print(f"✓ Extracted {len(texts)} conversation texts")
    
    np.random.seed(42)
    indices = np.random.permutation(len(texts))
    split_idx = int(len(texts) * train_ratio)
    
    train_texts = [texts[i] for i in indices[:split_idx]]
    val_texts = [texts[i] for i in indices[split_idx:]]
    
    print(f"\n✓ Split: {len(train_texts)} train, {len(val_texts)} validation")
    
    print("\n" + "-" * 60)
    print("Building vocabulary...")
    print("-" * 60)
    tokenizer = SimpleTokenizer(vocab_size=5000)
    tokenizer.build_vocab(train_texts)
    
    print("\n" + "-" * 60)
    print("Creating datasets...")
    print("-" * 60)
    train_dataset = ConversationDataset(train_texts, tokenizer, max_length=256)
    val_dataset = ConversationDataset(val_texts, tokenizer, max_length=256)
    
    print(f"\n" + "-" * 60)
    print("Sample training example:")
    print("-" * 60)
    input_ids, target_ids = train_dataset[0]
    print(f"Input tokens:  {input_ids.tolist()[:10]}... (length={len(input_ids)})")
    print(f"Target tokens: {target_ids.tolist()[:10]}... (length={len(target_ids)})")
    print(f"Decoded input: {tokenizer.decode(input_ids.tolist()[:20])}")
    
    return train_dataset, val_dataset, tokenizer


def collate_fn(batch, pad_idx=0):
    """Collate function to pad variable-length sequences."""
    input_ids = [item[0] for item in batch]
    target_ids = [item[1] for item in batch]
    
    max_len = max(len(seq) for seq in input_ids)
    
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(input_ids, target_ids):
        pad_len = max_len - len(inp)
        
        padded_inp = torch.cat([
            inp,
            torch.full((pad_len,), pad_idx, dtype=torch.long)
        ])
        
        padded_tgt = torch.cat([
            tgt,
            torch.full((pad_len,), pad_idx, dtype=torch.long)
        ])
        
        padded_inputs.append(padded_inp)
        padded_targets.append(padded_tgt)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)


def plot_training_curves(history: Dict):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_perplexity'], label='Train Perplexity', marker='o')
    plt.plot(history['val_perplexity'], label='Val Perplexity', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.axhline(y=50, color='r', linestyle='--', label='Target (< 50)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=150)
    print(f"✓ Training curves saved to: loss_curves.png")
    plt.close()


def train_model(model: SimpleTransformer, train_dataset, val_dataset, 
                tokenizer: SimpleTokenizer, epochs: int = 5, batch_size: int = 32):
    """Train the transformer model."""
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    model = model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_perplexity': [],
        'val_perplexity': []
    }
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print('='*60)
        
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if (batch_idx + 1) % 20 == 0:
                avg_loss = np.mean(train_losses[-20:])
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} | Loss: {avg_loss:.4f}")
        
        avg_train_loss = np.mean(train_losses)
        train_perplexity = np.exp(avg_train_loss)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                
                logits = model(input_ids)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        val_perplexity = np.exp(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_perplexity'].append(train_perplexity)
        history['val_perplexity'].append(val_perplexity)
        
        print(f"\n  Train Loss: {avg_train_loss:.4f} | Perplexity: {train_perplexity:.2f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
        
        # Generate sample
        if (epoch + 1) % 2 == 0:
            print(f"\n  Sample generation:")
            sample_text = model.generate(tokenizer, "i need", max_new_tokens=15, device=device)
            print(f"    Prompt: 'i need'")
            print(f"    Generated: {sample_text}")
    
    # Save model
    print(f"\n{'='*60}")
    print("Saving model...")
    print('='*60)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': len(tokenizer.word2idx),
        'history': history
    }, 'pretrained_model.pt')
    
    print(f"✓ Model saved to: pretrained_model.pt")
    
    plot_training_curves(history)
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print('='*60)
    print(f"  Final Train Perplexity: {history['train_perplexity'][-1]:.2f}")
    print(f"  Final Val Perplexity: {history['val_perplexity'][-1]:.2f}")
    
    if history['val_perplexity'][-1] < 50:
        print(f"  ✓ Target achieved! (Perplexity < 50)")
    else:
        print(f"  ⚠ Target not met (need < 50)")
    
    print(f"\n  Sample Generations:")
    prompts = ["i need", "what is", "can you", "the product"]
    for prompt in prompts:
        generated = model.generate(tokenizer, prompt, max_new_tokens=12, device=device)
        print(f"    '{prompt}' → {generated}")
    
    return model, history


if __name__ == "__main__":
    csv_path = "./cleaned_data.csv"
    
    try:
        train_dataset, val_dataset, tokenizer = load_and_prepare_data(csv_path)
        
        print("\n" + "=" * 60)
        print("✓ Data preparation complete!")
        print("=" * 60)
        print(f"Vocabulary size: {len(tokenizer.word2idx)}")
        print(f"Train sequences: {len(train_dataset)}")
        print(f"Val sequences: {len(val_dataset)}")
        
        print("\n" + "=" * 60)
        print("STEP 2: Building Transformer Model")
        print("=" * 60)
        
        model = SimpleTransformer(
            vocab_size=len(tokenizer.word2idx),
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            max_len=256
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n✓ Model created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Embedding dim: 128")
        print(f"  Num heads: 4")
        print(f"  Num layers: 2")
        
        print("\n" + "-" * 60)
        print("Testing forward pass...")
        print("-" * 60)
        
        sample_input, sample_target = train_dataset[0]
        sample_input = sample_input.unsqueeze(0)
        
        with torch.no_grad():
            logits = model(sample_input)
        
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {sample_input.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Expected: (batch=1, seq_len={sample_input.shape[1]}, vocab={len(tokenizer.word2idx)})")
        
        print("\n" + "=" * 60)
        print("STEP 3: Training the Model")
        print("=" * 60)
        
        train_model(model, train_dataset, val_dataset, tokenizer, epochs=5)
        
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find {csv_path}")
        print("Make sure cleaned_data.csv from Part 1 is in the parent directory!")