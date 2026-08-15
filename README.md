# BCG Sim2Real Deep Learning

本專案使用合成 BCG（Ballistocardiogram，心沖擊圖）訓練深度學習模型，從含呼吸與雜訊的 BCG 中重建心臟成分，並以真實 ECG 心率評估 Sim2Real 表現。

## 專案流程

1. `bcg_signal_synthesize.py` 產生 noisy/clean BCG 配對。
2. `train_BCG_HeartFilter.py` 使用雙向 LSTM 與 MSE + Morlet CWT loss 訓練降噪模型。
3. `validate_performance.py` 在真實 BCG/ECG 配對上比較原始與模型處理後的心率。
4. 評估結果包含逐筆 CSV、MAE、RMSE、bias、95% LoA、相關係數及 ±3/±5 BPM 命中率。

## 環境安裝

建議使用 Python 3.10–3.12：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 預設資料位置

程式會依專案位置推導以下預設路徑：

```text
../Dataset/BCG/Synthesis/training
../Dataset/BCG/Synthesis/validation
../Dataset/BCG/Synthesis/test
../Dataset/BCG/DeepLearningData/BCG_ECG_10sec
```

每個 `.npy` 檔案格式為 `(2, 1000)`：

- 合成資料：`[noisy_bcg, clean_bcg]`
- 真實資料：`[bcg, ecg]`
- 取樣率：100 Hz
- 訊號長度：10 秒

所有路徑皆可透過命令列參數覆寫。

## 產生合成資料

產生少量測試資料：

```bash
python bcg_signal_synthesize.py --count 100 --output-dir /path/to/Synthesis
```

預設以 `80% / 10% / 10%` 分割 training、validation 與 test；已存在的檔案不會被覆寫，除非加入 `--overwrite`。

## 訓練

```bash
python train_BCG_HeartFilter.py \
  --train-dir ../Dataset/BCG/Synthesis/training \
  --val-dir ../Dataset/BCG/Synthesis/validation \
  --output-dir weight/BCG_HeartFilter/my_run
```

輸出包含：

- `config.json`：本次完整設定
- `model_summary.txt`：模型摘要
- `checkpoint.pth`：可恢復 optimizer/scheduler 的 checkpoint
- `best_train_model.pth`：最佳平均訓練 loss 權重
- `best_val_model.pth`：最佳驗證 loss 權重
- `train.log`、`loss.png`

同一個輸出目錄預設會自動恢復 checkpoint；若要重新開始可加入 `--no-resume`。快速 smoke test 可使用：

```bash
python train_BCG_HeartFilter.py --epochs 1 --batch-size 2 \
  --max-train-samples 2 --max-val-samples 2 \
  --hidden-size 4 --num-layers 1 --output-dir /tmp/bcg_smoke --device cpu --no-resume
```

## 真實資料評估

使用現有預設權重與資料：

```bash
python validate_performance.py
```

或指定路徑及樣本數：

```bash
python validate_performance.py \
  --data-dir /path/to/BCG_ECG_10sec \
  --weights weight/BCG_HeartFilter/260109/best_Test_model.pth \
  --output-dir reports/real_world \
  --limit 100
```

報告會寫入 `metrics.json`、`per_sample.csv` 與兩張 Bland–Altman 圖。完整評估不會因單筆心率偵測失敗而停止；無效結果會記為 `NaN` 並從該指標的有效配對中排除。

### 傳統演算法 baseline

```bash
python evaluate_baselines.py
```

這會比較 FFT peak、autocorrelation 與 HeartV6，並輸出 overall、by-subject、by-distance 結果。ECG reference 使用帶品質診斷的 QRS detector，同時保留 `ecg_hr_legacy` 供稽核。受試者分布不均，因此研究結論不應只使用所有片段直接合併的 overall 指標；請同時檢查 `by_subject.csv`。

## Algorithm-aware residual filter

兩階段訓練：先以合成 noisy/clean BCG 做 supervised pretraining，再以 subject-disjoint 真實 BCG 和 ECG-derived HR 做 fine-tuning：

```bash
python train_algorithm_aware_filter.py \
  --fold 0 \
  --pretrain-epochs 20 \
  --finetune-epochs 10 \
  --output-dir weight/AlgorithmAwareFilter/fold_0
```

每個 fold 的 test subject 完全不參與梯度與 checkpoint selection。模型採 residual identity initialization；真實資料階段預設使用貼近 HeartV6 的 differentiable surrogate：六個重疊 Butterworth 零相位頻率響應、整流頻譜融合、候選諧波計分與 target-vs-distractor ranking，再加上 identity regularization 和 residual total variation。舊的單一 envelope objective 保留作 ablation：

```bash
python train_algorithm_aware_filter.py --real-objective envelope ...
```

目前小型 pilot（每 fold 1,000 筆 balanced real training segments）僅用於檢查研究方向，不能替代完整十折 LOSO。前兩個 held-out subjects 上，filtered HeartV6 的 MAE 分別由 1.3685 降至 1.2565 BPM、由 1.3450 降至 1.2980 BPM；Subject 2 的 raw HeartV7 仍略優於 filtered pipeline。完整結論需等待全部 folds、subject-cluster uncertainty 與 ablation。

快速 smoke test：

```bash
python train_algorithm_aware_filter.py \
  --pretrain-epochs 1 --finetune-epochs 1 --batch-size 2 \
  --channels 8 --blocks 2 \
  --max-synthetic-train 2 --max-synthetic-val 2 \
  --max-real-train 2 --max-real-val 2 \
  --output-dir /tmp/algorithm_aware_smoke --device cpu
```

## 測試

```bash
python -m unittest discover -s tests -v
python -m compileall -q . -x '(^|/)(.venv|.git)/'
```

## LaTeX 研究報告

先由最新評估結果重建圖表，再編譯論文：

```bash
python generate_paper_assets.py
cd paper
tectonic main.tex
```

原始稿位於 `paper/main.tex`，輸出為 `paper/main.pdf`。目前版本是研究進度稿；HeartV7 數值屬 exploratory development-set results，待 nested leave-one-subject-out 實驗後才能改寫為正式 generalization claim。

## 研究結果解讀

合成驗證 loss 只能說明模型學會合成資料的重建任務。Sim2Real 結論應以完全獨立、按受試者切分的真實測試資料為準，並同時報告傳統方法與模型方法的心率指標、失敗率及信賴區間。本專案不應直接作為醫療診斷工具。
