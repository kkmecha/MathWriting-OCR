import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

def convert_inkml_to_png(inkml_file, png_file):
    # InkMLファイルをパースする
    tree = ET.parse(inkml_file)
    root = tree.getroot()

    traces = []
    for trace in root.findall('{http://www.w3.org/2003/InkML}trace'):
        trace_data = trace.text.strip().split(',')
        points = [(float(x.split()[0]), float(x.split()[1])) for x in trace_data]
        traces.append(points)

    # トレースに基づいて画像サイズを決定
    all_x = [x for trace in traces for x, y in trace]
    all_y = [y for trace in traces for x, y in trace]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width, height = int(max_x - min_x) + 10, int(max_y - min_y) + 10

    # 新しい画像を作成
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # トレースを画像に描画
    for trace in traces:
        trace = [(x - min_x + 5, y - min_y + 5) for x, y in trace]
        draw.line(trace, fill='black', width=2)

    # PNGファイルとして画像を保存
    image.save(png_file)

def convert_png_to_grayscale(png_file, grayscale_file):
    # PNGファイルを開く
    image = Image.open(png_file)
    # 画像をグレースケールに変換
    grayscale_image = image.convert('L')
    # グレースケール画像を保存
    grayscale_image.save(grayscale_file)
