# tensor_chw とは

tensor_chw は、モデル入力に使う 3 次元テンソルです。

- C: チャネル数（特徴量の種類）
- H: 高さ方向のセル数
- W: 幅方向のセル数

このプロジェクトでは、tensor_chw は C × H × W の順番で保存されています。

## このリポジトリでの意味

- channels の i 番目の名前に対応する 2 次元マップが tensor_chw の i 番目です。
- つまり、channels と tensor_chw は同じ順序で 1 対 1 対応しています。

例:

- channels[0] = layer01_copper_ratio
- tensor_chw[0] = layer01_copper_ratio の空間マップ

## 今回の生成ファイルの形

fno_features.json では、次の形です。

- num_channels = 56
- grid_shape.height = 4
- grid_shape.width = 4
- よって tensor_chw の形は 56 × 4 × 4

## 中身の作られ方

build_fno_features.py では、次の順でチャネルを追加しています。

1. 各レイヤの copper_ratio（空間マップ）
2. 各レイヤのスカラー値を 2 次元にブロードキャストしたマップ
3. delta_temperature_C
4. 境界条件マスク（bc_ux_mask, bc_uy_mask, bc_uz_mask）
5. 座標チャネル（x_coord, y_coord）

## 学習時の扱い

この JSON は 1 ケース分です。
複数ケースをまとめると、一般に N × C × H × W（N はケース数）として扱います。

## 実務メモ

- channels の順番を変えると、学習済みモデルとの整合が崩れるので固定してください。
- 学習前にチャネルごとの正規化を行うと安定しやすくなります。
- shape チェック（C, H, W）が最初のデバッグポイントです。
