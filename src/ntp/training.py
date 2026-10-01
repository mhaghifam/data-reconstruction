import numpy as np
import torch
import torch.distributions as distributions
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


def train_epoch(model, dataloader, optimizer, device,scheduler=None, unknown_half=False, pad_value=-1):
    """One pass over the training data.

    unknown_half=True also trains the positions past each sequence's end, whose bits no training
    sample reveals: the padding there is replaced by uniformly random bits and the target is 1/2.
    Accuracy is always measured on the observed positions only.
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        input_seq = batch['input'].to(device)
        target_seq = batch['target'].to(device)
        lengths = batch['length'].to(device)
        loss_mask = batch['loss_mask'].to(device)

        if unknown_half:
            random_bits = torch.randint(0, 2, input_seq.shape, device=device)
            input_seq = torch.where(input_seq == pad_value, random_bits, input_seq)
            targets = torch.where(loss_mask > 0, target_seq.float(), torch.full_like(loss_mask, 0.5))
            weights = torch.ones_like(loss_mask)
        else:
            targets = target_seq.float()
            weights = loss_mask

        # Forward pass
        logits = model(input_seq, lengths)

        # Compute loss on the weighted positions (observed ones only, unless unknown_half)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        loss_all = loss_fn(logits, targets)
        loss = (loss_all * weights).sum() / weights.sum()

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
            

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens

    return avg_loss, accuracy

def evaluate_last_position(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    with torch.no_grad():
        for batch in dataloader:
            input_seq = batch['input'].to(device)
            target_seq = batch['target'].to(device)
            lengths = batch['length'].to(device)

            logits = model(input_seq)

            batch_size = input_seq.shape[0]
            last_positions = lengths - 1

            batch_indices = torch.arange(batch_size, device=device)
            last_logits = logits[batch_indices, last_positions]
            last_targets = target_seq[batch_indices, last_positions]

            # Compute loss at last position
            loss = loss_fn(last_logits, last_targets.float())
            total_loss += loss.sum().item()

            predictions = (last_logits > 0).long()
            correct = (predictions == last_targets).sum().item()

            total_correct += correct
            total_samples += batch_size

    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    return avg_loss, accuracy



def train(model, train_loader, val_loader, num_epochs, lr, device):
    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,          
    weight_decay=0.0  
    )

    
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=num_epochs * len(train_loader) // 2, 
        gamma=0.1
    )

    
    max_val_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scheduler)
        val_loss, val_acc = evaluate_last_position(model, val_loader, device)
        if val_acc>=max_val_acc:
            max_val_acc = val_acc



        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Vall Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"max Val Acc:   {max_val_acc:.4f}")

    return model
