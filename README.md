# FNO-study

FNO（Fourier Neural Operator）の学習用リポジトリ。大規模な解析結果を予測することを目指す。

---

## 概要 / Overview

このリポジトリは、偏微分方程式（PDE）のソリューション演算子を学習するための
**Fourier Neural Operator (FNO)** の実装です。

- 1次元問題（例: Burgersの方程式）
- 2次元問題（例: Navier-Stokes方程式）
- 3次元時空間問題

に対応した FNO モデルを提供します。

---

## ディレクトリ構成 / Repository Structure

```
FNO-study/
├── fno/                   # メインパッケージ
│   ├── __init__.py
│   ├── layers.py          # SpectralConv1d / 2d / 3d
│   ├── model.py           # FNO1d / FNO2d / FNO3d
│   ├── data.py            # データセット・データローダー
│   └── train.py           # トレーナー・損失関数
├── scripts/
│   └── train.py           # 学習スクリプト (CLI)
├── configs/
│   ├── burgers.yaml       # Burgers 方程式のデフォルト設定
│   └── navier_stokes.yaml # Navier-Stokes のデフォルト設定
├── tests/                 # pytest テスト
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

---

## インストール / Installation

```bash
pip install -r requirements-dev.txt
pip install -e .
```

---

## 使い方 / Usage

### Burgers方程式の学習（合成データ）

```bash
python scripts/train.py --problem burgers --n_epochs 100
```

### Navier-Stokes方程式の学習

```bash
python scripts/train.py --config configs/navier_stokes.yaml
```

### 独自データセットを使用する場合

`.npz` ファイルに `"a"` (入力) と `"u"` (出力) のキーで保存してください:

```bash
python scripts/train.py --problem burgers --data_path /path/to/burgers.npz
```

### YAML 設定ファイルを使用する場合

```bash
python scripts/train.py --config configs/burgers.yaml
```

---

## Python API

```python
from fno import FNO1d, FNO2d
from fno.data import BurgersDataset, make_dataloaders
from fno.train import Trainer

# モデルの定義
model = FNO1d(modes=16, width=64, in_channels=2, out_channels=1)

# データの準備
dataset = BurgersDataset(n_samples=1000, n_x=128)
train_loader, val_loader = make_dataloaders(dataset, batch_size=32)

# 学習
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=100,
    checkpoint_dir="checkpoints/burgers",
)
trainer.train()
```

---

## テスト / Testing

```bash
pytest tests/
```

---

## 参考文献 / References

- Z. Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021.
  [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
