def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        if batch.x.size(0) == 0:
            continue
        outputs = model(batch)
        targets = batch.y
        if targets.dim() == 3 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        elif targets.dim() == 2 and targets.size(1) == 5:
            targets = targets
        else:
            print(f"Unexpected target shape: {targets.shape}")
            continue
        if outputs.size() != targets.size():
            print(f"Size mismatch: outputs {outputs.size()}, targets {targets.size()}")
            continue
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if batch.x.size(0) == 0:
                continue
            outputs = model(batch)
            targets = batch.y
            if targets.dim() == 3 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            elif targets.dim() == 2 and targets.size(1) == 5:
                targets = targets
            else:
                print(f"Unexpected target shape: {targets.shape}")
                continue
            if outputs.size() != targets.size():
                print(f"Size mismatch: outputs {outputs.size()}, targets {targets.size()}")
                continue
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / max(num_batches, 1)

print("Training setup complete")

num_epochs = 100
best_val_loss = float('inf')
patience = 15
patience_counter = 0

print("Starting training...")

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate_epoch(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

print("Training completed")
print(f"Best validation loss: {best_val_loss:.4f}")

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

print("Model loaded with best weights")
