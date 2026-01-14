import numpy as np
import torch
import torch.distributions as distributions
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


def train_epoch(model, dataloader, optimizer, device,scheduler=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        input_seq = batch['input'].to(device)
        target_seq = batch['target'].to(device)
        lengths = batch['length'].to(device)
        loss_mask = batch['loss_mask'].to(device)

        # Forward pass
        logits = model(input_seq, lengths)

        # Compute loss only on valid positions
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        loss_all = loss_fn(logits, target_seq.float())

        # Mask out padding positions
        loss_masked = loss_all * loss_mask
        loss = loss_masked.sum() / loss_mask.sum()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Track metrics
        total_loss += loss.item() * loss_mask.sum().item()

        # Accuracy on valid positions
        predictions = (logits > 0).long()
        correct = ((predictions == target_seq) * loss_mask).sum().item()
        total_correct += correct
        total_tokens += loss_mask.sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens

    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_seq = batch['input'].to(device)
            target_seq = batch['target'].to(device)
            lengths = batch['length'].to(device)
            loss_mask = batch['loss_mask'].to(device)

            logits = model(input_seq, lengths)

            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            loss_all = loss_fn(logits, target_seq.float())
            loss_masked = loss_all * loss_mask
            loss = loss_masked.sum() / loss_mask.sum()

            total_loss += loss.item() * loss_mask.sum().item()

            predictions = (logits > 0).long()
            correct = ((predictions == target_seq) * loss_mask).sum().item()
            total_correct += correct
            total_tokens += loss_mask.sum().item()
            print(total_tokens)

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens

    return avg_loss, accuracy

def evaluate_last_position(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_seq = batch['input'].to(device)
            target_seq = batch['target'].to(device)
            lengths = batch['length'].to(device)  # move to device

            logits = model(input_seq)

            batch_size = input_seq.shape[0]
            last_positions = lengths - 1

            # Ensure indices are on correct device
            batch_indices = torch.arange(batch_size, device=device)
            last_logits = logits[batch_indices, last_positions]
            last_targets = target_seq[batch_indices, last_positions]

            predictions = (last_logits > 0).long()
            correct = (predictions == last_targets).sum().item()

            total_correct += correct
            total_samples += batch_size

    accuracy = total_correct / total_samples
    return accuracy



def train(model, train_loader, val_loader, num_epochs, lr, device):
    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-4,          # Slightly lower peak to be safe without heavy regularization
    weight_decay=0.0  # CRITICAL: Allow weights to grow to memorize data
    )

    # 2. Use a simpler scheduler (StepLR) or just constant LR with warmup
    # This ensures we don't decay too fast. We want to grind the loss to zero.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=num_epochs * len(train_loader) // 2, # Drop LR only halfway through
        gamma=0.1
    )

    # Inside train_epoch loop, after optimizer.step():
    max_val_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scheduler)
        val_acc = evaluate_last_position(model, val_loader, device)
        if val_acc>=max_val_acc:
            max_val_acc = val_acc



        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Acc:   {val_acc:.4f}")
        print(f"max Val Acc:   {max_val_acc:.4f}")

    return model
