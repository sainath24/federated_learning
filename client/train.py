from tqdm import tqdm
import torch
def train(model, train_loader, optimizer, criterion, epochs, device):
    model.to(device)
    model.train()    
    tr_loss = 0
    
    tk0 = tqdm(train_loader, desc="Iteration")

    for step, batch in enumerate(tk0):

        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        
        # Runs the forward pass with autocasting.
        # with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # scaler.scale(loss).backward()
        loss.backward()

        tr_loss += loss.item()

        # scaler.step(optimizer)
        optimizer.step()

        # Updates the scale for next iteration.
        # scaler.update()
        optimizer.zero_grad()
    return loss.item()