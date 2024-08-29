import os
from src.data_processing import convert_inkml_to_png, convert_png_to_grayscale
import config  # 設定ファイルのインポート

def convert_data(input_dir, output_png_dir, output_grayscale_dir):
    # 入力ディレクトリ内のすべてのInkMLファイルをPNGに変換
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.inkml'):
                inkml_file = os.path.join(root, file)
                png_file = os.path.join(output_png_dir, file.replace('.inkml', '.png'))
                convert_inkml_to_png(inkml_file, png_file)
                print(f'変換完了: {inkml_file} -> {png_file}')
    
    # PNG画像をグレースケールに変換
    for root, _, files in os.walk(output_png_dir):
        for file in files:
            if file.endswith('.png'):
                png_file = os.path.join(root, file)
                grayscale_file = os.path.join(output_grayscale_dir, file)
                convert_png_to_grayscale(png_file, grayscale_file)
                print(f'変換完了: {png_file} -> {grayscale_file}')

if __name__ == "__main__":
    input_dirs = {
        config.TRAIN_INKML_DIR: (config.TRAIN_PNG_DIR, config.TRAIN_GRAYSCALE_DIR),
        config.VALID_INKML_DIR: (config.VALID_PNG_DIR, config.VALID_GRAYSCALE_DIR),
        config.TEST_INKML_DIR: (config.TEST_PNG_DIR, config.TEST_GRAYSCALE_DIR)
    }

    for input_dir, (output_png_dir, output_grayscale_dir) in input_dirs.items():
        os.makedirs(output_png_dir, exist_ok=True)
        os.makedirs(output_grayscale_dir, exist_ok=True)
        convert_data(input_dir, output_png_dir, output_grayscale_dir)
