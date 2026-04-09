from tqdm import tqdm

def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)

        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)