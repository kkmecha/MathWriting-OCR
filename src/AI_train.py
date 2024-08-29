import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # PILからImageをインポート

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10クラス分類の場合

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(dataset_path, epochs=10, batch_size=32):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    dataset = CustomDataset(root_dir=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = SimpleCNN().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = len(dataloader)
        for batch_idx, inputs in enumerate(dataloader):
            inputs = inputs.cuda(non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = torch.randint(0, 10, (inputs.size(0),)).cuda(non_blocking=True)  # ダミーラベル
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            percent_complete = (batch_idx + 1) / total_batches * 100
            print(f"Training Model: Epoch {epoch+1}/{epochs}, {percent_complete:.2f}% complete ({batch_idx + 1}/{total_batches}), Loss: {running_loss / ((batch_idx + 1) * batch_size):.4f}")

        print(f"Epoch {epoch+1}/{epochs} completed. Loss: {running_loss / len(dataset):.4f}")
    print("Model training complete.")
    
    # モデルの保存
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model training complete and saved as 'trained_model.pth'.")

# メイン処理
if __name__ == "__main__":
    train_dataset_path = 'D:/Application/ocr/dataset/grayscale_images/'
    train_model(train_dataset_path)
