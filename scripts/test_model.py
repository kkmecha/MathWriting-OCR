import numpy as np
import torch
from PIL import Image
from src.model import SimpleCNN
import config

def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    return model

def test_model(test_image_path):
    model = load_model()
    image = Image.open(test_image_path).convert('L')
    image = torch.tensor(np.array(image)).unsqueeze(0).unsqueeze(0).float()  # Convert image to tensor
    with torch.no_grad():
        output = model(image)
    return output

if __name__ == "__main__":
    test_image_path = 'data/test/sample_image.png'  # テスト用画像パス
    output = test_model(test_image_path)
    print(f'モデル出力: {output}')
