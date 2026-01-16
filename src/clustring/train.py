import torch
import torch.nn as nn

def train(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda', verbose=True):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)

        train_losses.append(total_loss / len(train_loader))
        train_accs.append(correct / total)

        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total += y_batch.size(0)

        val_losses.append(total_loss / len(val_loader))
        val_accs.append(correct / total)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: train_loss={train_losses[-1]:.4f}, train_acc={train_accs[-1]:.4f}, val_loss={val_losses[-1]:.4f}, val_acc={val_accs[-1]:.4f}")

    return model, train_accs, train_losses, val_accs, val_losses