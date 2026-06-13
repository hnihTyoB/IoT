# 🔐 IoT Device Identification via Self-Supervised Learning

> **Nhận diện thiết bị IoT dựa trên hành vi bằng Self-Supervised Learning**

Hệ thống phân loại **26 thiết bị IoT** dựa trên hành vi traffic mạng, sử dụng kiến trúc **Transformer Encoder** kết hợp **Contrastive Learning** và **Masked Feature Modeling** — không cần label spoofing.

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
│ patterns without │ behavioral       │ 26 device classes         │
│ any labels       │ embeddings       │                           │
└──────────────────┴──────────────────┴───────────────────────────┘
         ↓                  ↓                     ↓
    Pre-trained        Contrastive            Classifier
    Encoder Weights    Encoder Weights        + t-SNE + Report
```

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
├── inference.py                  # Công cụ dự đoán (Inference Engine) cho thiết bị lạ/quen
├── IoT_Training_R2.ipynb         # Notebook huấn luyện & tích hợp Cloudflare R2 trên Colab
├── requirements.txt
└── README.md
```

## 🚀 Hướng dẫn huấn luyện trên Google Colab (Khuyên dùng)

Dự án cung cấp sẵn file notebook [IoT_Training_R2.ipynb](./IoT_Training_R2.ipynb) được thiết kế đặc biệt để chạy trên môi trường Google Colab (sử dụng GPU T4 miễn phí) và tự động sao lưu/khôi phục dữ liệu từ Cloudflare R2.

**Các bước thực hiện:**
1. Tải file [IoT_Training_R2.ipynb](./IoT_Training_R2.ipynb) lên Google Colab hoặc Google Drive của bạn.
2. Đảm bảo bạn đã chuẩn bị tài khoản Cloudflare R2 và tạo một bucket chứa file `dataset.zip` (hoặc upload thủ công lên Colab nếu không dùng R2).
3. Mở notebook trên Colab, chọn Runtime là **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
4. Điền các thông tin xác thực R2 của bạn ở Cell số 2:
   ```python
   R2_ACCOUNT_ID = "nhap_account_id_cua_ban"
   R2_ACCESS_KEY = "nhap_access_key"
   R2_SECRET_KEY = "nhap_secret_key"
   BUCKET_NAME = "ten-bucket-cua-ban"
   ```
5. Chạy tuần tự các cell của notebook. Notebook sẽ tự động:
   * Cài đặt thư viện cần thiết.
   * Kết nối Cloudflare R2 để tải dữ liệu huấn luyện và các checkpoints mô hình cũ (nếu có).
   * Chạy huấn luyện (Phase 3 Finetune hoặc các Phase khác nếu tùy chỉnh).
   * Đóng gói kết quả huấn luyện dạng `.zip` và tải ngược trở lại lên Cloudflare R2.

---

## 💻 Hướng dẫn huấn luyện trên máy tính cá nhân (Dành cho test/debug)

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Tải bộ dataset từ [link Google Drive này](https://drive.google.com/file/d/1ALsVGOSsNgeNA502QJhKw55Qr-ril13p/view?usp=sharing).
Sau khi giải nén, đặt 26 file CSV từ UNSW-IoTraffic vào thư mục `../dataset/` (thư mục này nằm ngang hàng với thư mục mã nguồn dự án):

```
dataset/
├── AmazonEcho_44650d56ccd3_flows.csv
├── AugustDoorBell_e076d03f00ae_flows.csv
├── ...
└── iHome_74c63b29d71d_flows.csv
```

### 3. Chạy từng Phase huấn luyện

```bash
# Phase 1: Pre-training (Masked Feature Modeling — ET-BERT style)
python run_experiments.py --config configs/pretrain_unsw.yaml

# Phase 2: Contrastive Learning (AOC-IDS style)
python run_experiments.py --config configs/contrastive_unsw.yaml

# Phase 3: Fine-tuning (Supervised Classification)
python run_experiments.py --config configs/finetune_unsw.yaml
```

### 4. Kết quả đầu ra

Mỗi thử nghiệm sẽ tự động tạo một thư mục tương ứng trong thư mục `experiments/` chứa các tệp:
- `model.pt` — Trọng số của mô hình đã huấn luyện
- `scaler_features.pkl` — Bộ chuẩn hóa dữ liệu
- `label_map.json` — Ánh xạ tên thiết bị sang nhãn số
- `device_dist_stats.json` — Khoảng cách và ngưỡng nhận diện lỗi cho từng thiết bị
- `centroids.pt` — Điểm trung tâm (centroid) của các thiết bị để phát hiện thiết bị lạ
- `training_curves.png` — Biểu đồ quá trình huấn luyện
- `tsne_*.png` — Biểu đồ phân tích separability bằng t-SNE
- `confusion_matrix.png` — Ma trận nhầm lẫn (ở Phase 3)
- `classification_report.txt` — Báo cáo phân loại chi tiết (F1-score, Precision, Recall)
- `config.yaml` — Bản sao cấu hình đã sử dụng

---

## 🔍 Hướng dẫn chạy thử nghiệm (Inference)

Để kiểm tra chương trình và nhận diện thiết bị từ file CSV traffic mạng, bạn có thể thực hiện chạy thử nghiệm theo 2 cách:

### Cách 1: Chạy trực tiếp trong Jupyter Notebook (Khuyên dùng)
Nếu bạn đang làm việc trên file notebook `.ipynb`, bạn có thể gọi trực tiếp API của class `IoTInferenceEngine` trong [inference.py](./inference.py) để tránh việc nhập liệu thủ công gây treo cell:

```python
from inference import IoTInferenceEngine, format_result

# Khởi tạo công cụ dự đoán với thư mục chứa kết quả của mô hình tốt nhất
engine = IoTInferenceEngine("experiments/finetune_robust_frozen")

# Chỉ định file traffic test cần kiểm tra (ví dụ trong demo_data/known)
test_file = "demo_data/known/test_AmazonEcho.csv"

# Thực hiện dự đoán hành vi và hiển thị kết quả
res = engine.predict(test_file)
print(format_result(res))
```

### Cách 2: Chạy qua giao diện Terminal/Dòng lệnh (Interactive)
Mở terminal và kích hoạt chế độ encoding UTF-8 (để tránh lỗi hiển thị tiếng Việt trên Windows) rồi chạy file script:

**Trên Windows PowerShell:**
```powershell
$env:PYTHONIOENCODING="utf-8"
python inference.py
```

**Trên CMD / Linux / macOS:**
```bash
python -X utf8 inference.py
```

Hệ thống sẽ hiển thị lời chào và yêu cầu bạn nhập đường dẫn file CSV traffic để nhận diện. Nhập đường dẫn bất kỳ (ví dụ: `demo_data/known/test_AmazonEcho.csv`) để xem phân tích, và gõ `q` để thoát chương trình.


## 📊 Features sử dụng (17 trường từ UNSW-IoTraffic)

| Feature                           | Ý nghĩa                       | Đóng góp vào nhận diện |
| --------------------------------- | ----------------------------- | ---------------------- |
| `srcAvgPayloadSize`               | Kích thước payload trung bình | Đặc trưng giao thức    |
| `srcAvgInterarrivalTime`          | Thời gian giữa các gói tin    | Chu kỳ gửi dữ liệu     |
| `srcStdDevPayloadSize`            | Độ lệch chuẩn payload         | Tính ổn định hành vi   |
| `flowDuration`                    | Thời lượng flow               | Pattern kết nối        |
| `srcNumPackets` / `dstNumPackets` | Số lượng gói tin              | Cường độ traffic       |
| ...                               | _(17 features tổng cộng)_     |                        |

## 🔬 Giải thích kỹ thuật

### Tại sao Contrastive Learning giải quyết được vấn đề "cùng giao thức"?

> _"Làm sao phân biệt Amazon Echo và Awair khi cả hai đều dùng NTP?"_

**Trả lời**: Dù cùng dùng NTP, mỗi thiết bị có **behavioral fingerprint** riêng:

- Amazon Echo: `srcAvgInterarrivalTime ≈ 27012ms`, `srcAvgPayloadSize ≈ 149 bytes`
- Awair: IAT và payload size khác biệt rõ rệt

CRC Loss bắt mô hình:

1. **Đưa** 2 chuỗi flow của Amazon Echo **lại gần nhau** (Positive Pair)
2. **Đẩy** chuỗi flow của Awair **ra xa** (Negative Pair)

### Pipeline 3 giai đoạn

```
Raw CSV Flows → [Windowing] → [Phase 1: MLM] → [Phase 2: Contrastive] → [Phase 3: Classify]
                  10 flows       Học patterns      Học separability        26 devices
                  per window     không cần label    behavioral embedding    classification
```

## ⚙️ Tùy chỉnh

Chỉnh sửa trong file YAML config:

| Parameter     | Mặc định | Mô tả                                 |
| ------------- | -------- | ------------------------------------- |
| `window_size` | 10       | Số flow liên tiếp trong 1 window      |
| `stride`      | 5        | Bước trượt giữa các window            |
| `d_model`     | 128      | Chiều Transformer embedding           |
| `nhead`       | 4        | Số attention heads                    |
| `num_layers`  | 4        | Số Transformer encoder layers         |
| `mask_ratio`  | 0.15     | Tỷ lệ feature bị mask (Phase 1)       |
| `temperature` | 0.1      | Temperature cho contrastive loss      |
| `loss_type`   | ntxent   | Loại loss: `ntxent`, `crc`, `triplet` |
