import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from src.model import SimpleCNN
import config
import torch.nn as nn
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        image = torch.tensor(np.array(image)).unsqueeze(0).float()  # Convert image to tensor
        label = self.labels[idx]
        return image, label

def train_model(train_loader, model, criterion, optimizer):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def main():
    # データセットとデータローダーの準備
    train_image_paths = [...]  # トレーニング画像パスのリスト
    train_labels = [...]  # トレーニングラベルのリスト
    train_dataset = CustomDataset(train_image_paths, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # モデル、損失関数、オプティマイザの初期化
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # モデルのトレーニング
    train_model(train_loader, model, criterion, optimizer)

    # モデルの保存
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f'モデルが {config.MODEL_SAVE_PATH} に保存されました。')

if __name__ == "__main__":
    main()
