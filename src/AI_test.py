import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from AI_train import SimpleCNN

# AIモデルの定義
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64 * 8 * 8, 128)
#         self.fc2 = nn.Linear(128, 10)  # 10クラス分類を想定

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool(x)
#         x = torch.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.view(-1, 64 * 8 * 8)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# モデルの評価
def evaluate_model(model_path, test_data_path):
    # モデルのロード
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # テストデータの前処理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # グレースケールに変換
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 正規化
    ])

    # テストデータの読み込み
    test_dataset = ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    # 推論の実行
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 精度の表示
    print(f'Accuracy of the model on the test images: {100 * correct / total}%')

if __name__ == "__main__":
    model_save_path = 'D:/Application/ocr/src/trained_model.pth'  # 保存したモデルのパス
    test_data_path = 'D:/Application/ocr/dataset/test/grayscale_images/'  # テストデータのパス

    evaluate_model(model_save_path, test_data_path)
