"""
defines the training loop.
"""

from tqdm import tqdm
import torch


def classification_train(model, train_loader, optimizer, criterion, epochs, device):
    model.to(device)
    model.train()
    tr_loss = 0

    tk0 = tqdm(train_loader, desc="Iteration", position=0, leave=True)

    for step, batch in enumerate(tk0):
        optimizer.zero_grad()

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
    return tr_loss.item()

def train_last_uga(initial_model_weights, model, train_loader, optimizer, criterion, epochs, device):
    model.to(device)
    model.train()
    tr_loss = 0

    tk0 = tqdm(train_loader, desc="Iteration", position=0, leave=True)

    for step, batch in enumerate(tk0):
        optimizer.zero_grad()

        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        # Runs the forward pass with autocasting.
        # with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # UGA STUFF
        temp_state_dict = model.state_dict()
        model.load_state_dict(initial_model_weights)

        outputs = model(inputs)
        loss_with_graph = criterion(outputs, labels)

        loss_with_graph.data = loss.data
        loss_with_graph.backward()

        model.load_state_dict(temp_state_dict)

        tr_loss += loss.item()

        # scaler.step(optimizer)
        optimizer.step()

        # Updates the scale for next iteration.
        # scaler.update()
        
    return tr_loss.item()