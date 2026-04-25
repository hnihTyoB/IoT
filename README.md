# 🔐 IoT Device Identification via Self-Supervised Learning

> **Nhận diện thiết bị IoT dựa trên hành vi bằng Self-Supervised Learning**

Hệ thống phân loại **27 thiết bị IoT** dựa trên hành vi traffic mạng, sử dụng kiến trúc **Transformer Encoder** kết hợp **Contrastive Learning** và **Masked Feature Modeling** — không cần label spoofing.

## 📐 Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                    IoT Device Identification                     │
├──────────────────┬──────────────────┬───────────────────────────┤
│   Phase 1        │   Phase 2        │   Phase 3                 │
│   ET-BERT Style  │   AOC-IDS Style  │   Classification          │
│   (Backbone)     │   (SSL Logic)    │   (Fine-tuning)           │
├──────────────────┼──────────────────┼───────────────────────────┤
│ Masked Feature   │ Contrastive      │ Supervised Cross-Entropy  │
│ Modeling (MLM)   │ Learning         │ Device Classification     │
│                  │ (CRC/NT-Xent)    │                           │
│ Learns traffic   │ Learns separable │ Maps embeddings to        │
│ patterns without │ behavioral       │ 27 device classes         │
│ any labels       │ embeddings       │                           │
└──────────────────┴──────────────────┴───────────────────────────┘
         ↓                  ↓                     ↓
    Pre-trained        Contrastive            Classifier
    Encoder Weights    Encoder Weights        + t-SNE + Report
```

## 🧩 Nguồn gốc & Tích hợp

| Component | Source | Vai trò | Tích hợp |
|-----------|--------|---------|----------|
| **Transformer Encoder** | bandwidth-estimation | "Cánh tay" — Pipeline | `models.py`: Sinusoidal PE, Attention Pooling, Pre-LN encoder |
| **Masked Feature Modeling** | ET-BERT | "Xương sống" — Backbone | `models.py`: Mask CSV features, reconstruct from context |
| **CRC / NT-Xent Loss** | AOC-IDS (Infocom 2024) | "Linh hồn" — SSL Logic | `losses.py`: Contrastive losses for behavioral embedding |
| **Flow Windowing** | bandwidth-estimation | Data pipeline | `dataset.py`: Sliding window over flow sequences |

## 📁 Cấu trúc dự án

```
iot-device-identification/
├── configs/
│   ├── pretrain_unsw.yaml        # Phase 1: Masked Feature Modeling
│   ├── contrastive_unsw.yaml     # Phase 2: Contrastive Learning
│   └── finetune_unsw.yaml        # Phase 3: Supervised Classification
├── functions_and_modules/
│   ├── __init__.py
│   ├── dataset.py                # IoTFlowWindowDataset, IoTContrastiveDataset
│   ├── models.py                 # IoTTransformerEncoder, MaskedFeatureModeling
│   ├── losses.py                 # CRCLoss, NTXentLoss, TripletLoss
│   ├── training.py               # pretrain_masked, train_contrastive, finetune
│   ├── visualization.py          # t-SNE, confusion matrix, training curves
│   └── experiment_artifacts.py   # Save/load experiment results
├── run_experiments.py            # Main entry point
├── requirements.txt
└── README.md
```

## 🚀 Cách chạy

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đặt 27 file CSV từ UNSW-IoTraffic (flows) vào thư mục `../dataset/`:
```
dataset/
├── AmazonEcho_44650d56ccd3_flows.csv
├── AugustDoorBell_e076d03f00ae_flows.csv
├── ...
└── iHome_74c63b29d71d_flows.csv
```

### 3. Chạy từng Phase

```bash
# Phase 1: Pre-training (Masked Feature Modeling — ET-BERT style)
python run_experiments.py --config configs/pretrain_unsw.yaml

# Phase 2: Contrastive Learning (AOC-IDS style)
python run_experiments.py --config configs/contrastive_unsw.yaml

# Phase 3: Fine-tuning (Supervised Classification)
python run_experiments.py --config configs/finetune_unsw.yaml
```

### 4. Kết quả

Mỗi experiment tạo thư mục trong `experiments/` chứa:
- `model.pt` — Trọng số mô hình
- `training_curves.png` — Biểu đồ training
- `tsne_*.png` — Phân tích separability bằng t-SNE
- `confusion_matrix.png` — Ma trận nhầm lẫn (Phase 3)
- `classification_report.txt` — Báo cáo phân loại chi tiết
- `config.yaml` — Cấu hình đã dùng

## 📊 Features sử dụng (17 trường từ UNSW-IoTraffic)

| Feature | Ý nghĩa | Đóng góp vào nhận diện |
|---------|---------|----------------------|
| `srcAvgPayloadSize` | Kích thước payload trung bình | Đặc trưng giao thức |
| `srcAvgInterarrivalTime` | Thời gian giữa các gói tin | Chu kỳ gửi dữ liệu |
| `srcStdDevPayloadSize` | Độ lệch chuẩn payload | Tính ổn định hành vi |
| `flowDuration` | Thời lượng flow | Pattern kết nối |
| `srcNumPackets` / `dstNumPackets` | Số lượng gói tin | Cường độ traffic |
| ... | *(17 features tổng cộng)* | |

## 🔬 Giải thích kỹ thuật

### Tại sao Contrastive Learning giải quyết được vấn đề "cùng giao thức"?

> *"Làm sao phân biệt Amazon Echo và Awair khi cả hai đều dùng NTP?"*

**Trả lời**: Dù cùng dùng NTP, mỗi thiết bị có **behavioral fingerprint** riêng:
- Amazon Echo: `srcAvgInterarrivalTime ≈ 27012ms`, `srcAvgPayloadSize ≈ 149 bytes`
- Awair: IAT và payload size khác biệt rõ rệt

CRC Loss bắt mô hình:
1. **Đưa** 2 chuỗi flow của Amazon Echo **lại gần nhau** (Positive Pair)
2. **Đẩy** chuỗi flow của Awair **ra xa** (Negative Pair)

### Pipeline 3 giai đoạn

```
Raw CSV Flows → [Windowing] → [Phase 1: MLM] → [Phase 2: Contrastive] → [Phase 3: Classify]
                  10 flows       Học patterns      Học separability        27 devices
                  per window     không cần label    behavioral embedding    classification
```

## ⚙️ Tùy chỉnh

Chỉnh sửa trong file YAML config:

| Parameter | Mặc định | Mô tả |
|-----------|----------|-------|
| `window_size` | 10 | Số flow liên tiếp trong 1 window |
| `stride` | 5 | Bước trượt giữa các window |
| `d_model` | 128 | Chiều Transformer embedding |
| `nhead` | 4 | Số attention heads |
| `num_layers` | 4 | Số Transformer encoder layers |
| `mask_ratio` | 0.15 | Tỷ lệ feature bị mask (Phase 1) |
| `temperature` | 0.1 | Temperature cho contrastive loss |
| `loss_type` | ntxent | Loại loss: `ntxent`, `crc`, `triplet` |
