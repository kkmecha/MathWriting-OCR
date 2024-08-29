[https://github.com/google-research/google-research/tree/master/mathwriting]

実行手順
データ変換

convert_data.py を実行して、InkMLファイルをPNGに変換し、そのPNGファイルをグレースケール画像に変換します。

python convert_data.py

モデルのトレーニング

train_model.py を実行して、トレーニングデータを使用してモデルをトレーニングします。

python train_model.py

モデルのテスト

test_model.py を実行して、保存されたモデルを使用してテスト画像の推論を行います。

python test_model.py
