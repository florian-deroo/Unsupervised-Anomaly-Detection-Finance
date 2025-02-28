import torch
from torch.utils.tensorboard import SummaryWriter

def train_model(model, epochs, train_loader, test_loader, optimizer, device, folder):
    writer = SummaryWriter(folder + 'runs/')

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = torch.nn.MSELoss()(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Log detailed metrics
        writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)

        if epoch % 1 == 0:
            val_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    reconstructed = model(batch)
                    val_loss += torch.nn.MSELoss()(reconstructed, batch).item()

            writer.add_scalar('Loss/val', val_loss / len(test_loader), epoch)

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'transformer_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': total_loss / len(train_loader),
                'val_loss': val_loss / len(test_loader),
            }, f'{folder}/checkpoints/model_epoch_{epoch}.pt')

    writer.close()