# FNO-study（日本語）

FNO 用のケース YAML を読み込み、検証し、モデル入力に使える空間チャネルへ変換するためのシンプルなユーティリティ集です。

## 実装済みの内容

- `examples/fno_input_case_example.yaml` からのケース読み込み
- キー項目とテンソル形状の基本バリデーション
- FNO 入力チャネルの構築（層マップ + グローバルチャネル）
- チャネルテンソルとメタデータの JSON 出力 CLI

## クイックスタート

1. 依存関係をインストール

```bash
pip install pyyaml
```

2. サンプルから特徴量を生成

PowerShell:

```powershell
python build_fno_features.py --input examples/fno_input_case_example.yaml --output artifacts/fno_features.json
```

Bash:

```bash
python build_fno_features.py \
  --input examples/fno_input_case_example.yaml \
  --output artifacts/fno_features.json
```

3. 生成された `artifacts/fno_features.json` を確認

## 補足

- 現在の実装はサンプルのデモマップ（`4x4`）を対象にしています。
- YAML に境界条件マスクがある場合は入力チャネルへ含めます。

## 次のステップ: 学習データセット作成

上面変位場（グリッド化済み）を学習させるために、FrontISTR の結果から target JSON を作成し、feature/target のペアをデータセット化します。

表面点群エクスポートから target JSON を生成:

```powershell
python build_fno_target_from_surface_points.py --feature-json artifacts/fno_features.json --surface-points examples/fno_surface_points_example.csv --output artifacts/fno_surface_target.json
```

表面点群サンプル: `examples/fno_surface_points_example.csv`

生成される target: `artifacts/fno_surface_target.json`

マニフェスト例: `examples/fno_training_manifest_example.json`

データセットを生成:

```powershell
python build_fno_training_dataset.py --manifest examples/fno_training_manifest_example.json --output artifacts/fno_dataset.npz --metadata-output artifacts/fno_dataset_metadata.json
```

生成物:

- `artifacts/fno_dataset.npz`（`inputs` と `targets`）
- `artifacts/fno_dataset_metadata.json`（shape、channels、case ID）

実務メモ:

- FrontISTR の生 AVS 出力しかない場合は、まず上面節点値を x, y, uz を含む表形式に出力してください。
- スクリプトが点群を特徴量グリッドへ補間して target JSON を作成します。

AVS/UCD 直読パス:

```powershell
python build_fno_target_from_avs_ucd.py --feature-json artifacts/fno_features.json --avs-ucd examples/frontistr_ucd_example.txt --output artifacts/fno_surface_target_from_avs_ucd.json --value-field uz
```

AVS/UCD サンプル: `examples/frontistr_ucd_example.txt`

この方法は ASCII 形式の AVS/UCD から節点データを直接読み、max z で上面を抽出して特徴量メッシュへ格子化します。

## FNO 学習

生成済みデータセットから直接学習:

```powershell
python train_fno.py --dataset artifacts/fno_dataset.npz --output-dir artifacts/training --split-file artifacts/splits/train_val_test_split.json --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 --epochs 200 --batch-size 8 --learning-rate 1e-3 --early-stopping-patience 20 --scheduler-patience 8
```

出力:

- `artifacts/training/fno_best.pt`（ベストチェックポイント）
- `artifacts/training/training_history.json`（エポックごとの指標）
- `artifacts/training/training_summary.json`（実行サマリ）

メモ:

- train/val/test 分割は `--split-file` で固定されます。
- split ファイルが無ければ初回に作成し、以降は再利用します。
- Early Stopping と ReduceLROnPlateau スケジューラはデフォルト有効です。

## 推論（1ケース）

学習済みチェックポイントで 1 ケースの予測場を出力:

```powershell
python infer_fno.py --checkpoint artifacts/training/fno_best.pt --dataset artifacts/fno_dataset.npz --case-index 0 --output artifacts/prediction_case.json
```

出力:

- `artifacts/prediction_case.json`（予測場。target があれば比較指標も含む）

## ベースライン評価

学習済みモデルのサマリーから主要指標を表示:

```powershell
python eval_baseline.py --summary artifacts/training/training_summary.json --history artifacts/training/training_history.json
```

コンソールに出力される内容:
- 学習設定（データセット、エポック数、学習率、モデル幅/深さ）
- 最良バリデーション MSE とテスト MSE/MAE
- 学習進捗（初期値 vs 最終値のロス）

## 予測場の可視化と誤差マップ

3 列並列プロット（正解 | 予測 | 誤差）を生成:

```powershell
python compare_predictions.py --checkpoint artifacts/training/fno_best.pt --dataset artifacts/fno_dataset.npz --num-cases 4 --output-dir artifacts/comparison
```

出力:
- 各ケースの PNG プロット（3 列レイアウト）
- `artifacts/comparison/comparison_summary.json`（ケースごとの MAE、RMSE、最大誤差）

必須: `pip install matplotlib`

## ハイパーパラメータ探索

FNO の幅、深さ、学習率で グリッドサーチを実行:

```powershell
python hp_search.py --dataset artifacts/fno_dataset.npz --split-file artifacts/splits/train_val_test_split.json --epochs 50 --device auto
```

6 つの設定を検証し出力:
- `artifacts/hp_search_report.json`（全結果と最適設定）
- `artifacts/hp_search/` 配下の設定ごとチェックポイント

探索範囲: width [16, 32, 64] × depth [2, 4, 6] × learning_rate [1e-4, 1e-3, 1e-2]

## 物理的に意味のある特徴量（銅密度マップ）

モデル解釈性を高めるため、空間を考慮した銅密度マップを生成:

```powershell
# 放射状パターン（中心が高密度）
python generate_copper_density_map.py --height 4 --width 4 --pattern radial --center-density 0.8 --edge-density 0.2 --output artifacts/copper_density_radial.json

# 縞状パターン（配線パターンをモデル）
python generate_copper_density_map.py --height 4 --width 4 --pattern striped --high-density 0.8 --low-density 0.2 --output artifacts/copper_density_striped.json
```

サポートパターン: `uniform`、`radial`、`striped`、`corner`

統合方法: YAML に密度マップを参照させてから特徴量を再構築:

```powershell
python build_fno_features.py --input examples/fno_input_case_enhanced.yaml --output artifacts/fno_features_enhanced.json
```

## 複数ケース一括推論と評価レポート自動生成

複数のテストケースで推論を実行、マークダウン要約レポートを生成:

```powershell
python batch_infer_fno.py --checkpoint artifacts/training/fno_best.pt --dataset artifacts/fno_dataset.npz --split-type test --output-dir artifacts/batch_inference
```

出力:
- `artifacts/batch_inference/batch_inference_summary.json`（集計メトリクス）
- `artifacts/batch_inference/report.md`（ケースごとメトリクス表）

ケースごと指標: MAE、RMSE、最大誤差、R²
集計指標: 全ケースの平均値

## 完全な一連のワークフロー

ベースライン確立から最適化までの全体的な流れについては [WORKFLOW.md](WORKFLOW.md) を参照してください。

## README 同期ポリシー

英語版と日本語版の差分を防ぐため、README は常にペアで更新します。

- `README.md` と `README_ja.md` は同一コミットで更新する。
- セクション構成とコマンド例を揃える。
- コミット前に [README_SYNC_CHECKLIST.md](README_SYNC_CHECKLIST.md) で確認する。

Sync Version: 2026-06-14（5つの新しい評価スクリプトで更新）
