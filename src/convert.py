import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor
import time
import cupy as cp  # CuPyのインポートを追加

# InkMLをPNGに変換する関数
def find_min_max_coordinates(trace_text):
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    points = trace_text.strip().split(',')
    for point in points:
        coords_pair = point.split()
        if len(coords_pair) >= 2:
            x, y = map(float, coords_pair[:2])
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    
    return min_x, max_x, min_y, max_y

def inkml_to_png(inkml_file, png_file, scale_factor=1.0):
    try:
        tree = ET.parse(inkml_file)
        root = tree.getroot()

        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
        for trace in root.findall('.//{http://www.w3.org/2003/InkML}trace'):
            x_min, x_max, y_min, y_max = find_min_max_coordinates(trace.text)
            min_x = min(min_x, x_min)
            max_x = max(max_x, x_max)
            min_y = min(min_y, y_min)
            max_y = max(max_y, y_max)

        width = int((max_x - min_x) * scale_factor)
        height = int((max_y - min_y) * scale_factor)

        image = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(image)

        for trace in root.findall('.//{http://www.w3.org/2003/InkML}trace'):
            points = trace.text.strip().split(',')
            coords = []
            for point in points:
                coords_pair = point.split()
                if len(coords_pair) >= 2:
                    x, y = map(float, coords_pair[:2])
                    coords.append((x * scale_factor - min_x * scale_factor,
                                   y * scale_factor - min_y * scale_factor))
            if coords:
                draw.line(coords, fill=0, width=2)

        image.save(png_file)
    except Exception as e:
        print(f"Error processing file {inkml_file}: {e}")

def convert_inkml_directory(input_dir, output_dir, scale_factor=2.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = [f for f in os.listdir(input_dir) if f.endswith('.inkml')]
    total_files = len(files)
    
    def process_file(filename):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.replace('.inkml', '.png'))
        inkml_to_png(input_file, output_file, scale_factor)
    
    with ThreadPoolExecutor() as executor:
        start_time = time.time()
        for idx, _ in enumerate(executor.map(process_file, files)):
            elapsed_time = time.time() - start_time
            percent_complete = (idx + 1) / total_files * 100
            print(f"Processing InkML files: {percent_complete:.2f}% complete ({idx + 1}/{total_files})")
    print("InkML to PNG conversion completed.")

# GPUでの画像変換処理
def process_image_gpu(image_path, output_path, scale_factor=1.0):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_cp = cp.asarray(image)

        new_size = (int(image_cp.shape[1] * scale_factor), int(image_cp.shape[0] * scale_factor))
        resized_image_cp = cp.array(cv2.resize(cp.asnumpy(image_cp), new_size, interpolation=cv2.INTER_LINEAR))

        result_image = cp.asnumpy(resized_image_cp).astype(np.uint8)
        cv2.imwrite(output_path, result_image)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def convert_images_to_grayscale_parallel(input_dir, output_dir, scale_factor=1.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    total_files = len(files)
    
    def process_file(filename):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        process_image_gpu(input_file, output_file, scale_factor)
    
    with ThreadPoolExecutor() as executor:
        start_time = time.time()
        for idx, _ in enumerate(executor.map(process_file, files)):
            elapsed_time = time.time() - start_time
            percent_complete = (idx + 1) / total_files * 100
            print(f"Processing Grayscale images: {percent_complete:.2f}% complete ({idx + 1}/{total_files})")
    print("Grayscale image conversion completed.")

# メイン処理
if __name__ == "__main__":
    input_dir = 'D:/Application/ocr/dataset/mathwriting-2024/test/'
    output_dir = 'D:/Application/ocr/dataset/test/png_images/'
    grayscale_output_dir = 'D:/Application/ocr/dataset/test/grayscale_images/'
    
    # train -> input_dir = 'D:/Application/ocr/dataset/mathwriting-2024/train/'
    #          output_dir = 'D:/Application/ocr/dataset/train/png_images/'
    #          grayscale_output_dir = 'D:/Application/ocr/dataset/train/grayscale_images/'
    
    # test ->  input_dir = 'D:/Application/ocr/dataset/mathwriting-2024/test/'
    #          output_dir = 'D:/Application/ocr/dataset/test/png_images/'
    #          grayscale_output_dir = 'D:/Application/ocr/dataset/test/grayscale_images/'
    
    scale_factor = 2.0

    convert_inkml_directory(input_dir, output_dir, scale_factor)
    convert_images_to_grayscale_parallel(output_dir, grayscale_output_dir, scale_factor)
