import torch
import torch.nn as nn
import torch.optim as optim
from models.net import RookstarNet
from train.data_loader import load_training_data


def train_value_head(epochs=5, batch_size=32, lr=1e-3):
    model = RookstarNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X, y = load_training_data()
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            _, value_pred = model(batch_X)
            loss = loss_fn(value_pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "models/weights/latest.pt")
    print("Model saved to models/weights/latest.pt")


if __name__ == '__main__':
    train_value_head()
