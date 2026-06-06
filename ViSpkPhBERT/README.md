# ViSPhB: Vietnamese Sentiment classification with Spiking neural networks and PhoBERT

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

**Phân loại cảm xúc tiếng Việt sử dụng Spiking Neural Networks và Knowledge Distillation từ PhoBERT**

[Giới thiệu](#giới-thiệu) • [Cài đặt](#cài-đặt) • [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng) • [Cấu trúc Project](#cấu-trúc-project)

</div>

---

## 📌 Giới thiệu

ViSPhB là một đồ án tốt nghiệp về **phân loại cảm xúc (sentiment analysis)** tiếng Việt kết hợp:

- **PhoBERT**: Mô hình Transformer được huấn luyện trước cho tiếng Việt, sử dụng làm giáo viên (teacher)
- **Spiking Neural Networks (SNNs)**: Mạng nơ-ron xung tương tự cấu trúc não bộ, tiết kiệm năng lượng hơn
- **Knowledge Distillation (KD)**: Chuyển giao kiến thức từ PhoBERT sang SNN để tối ưu hóa hiệu suất

### ✨ Điểm nổi bật

- ✅ Hỗ trợ phân loại cảm xúc **3 lớp** (Âm, Trung tính, Dương)
- ✅ Huấn luyện **đa giai đoạn** (Two-stage) và huấn luyện **trực tiếp** (Direct)
- ✅ Các biến thể: Tier 1 SpikeBERT, Tier 2 Implicit, Tier 3 Hybrid
- ✅ Demo tương tác với **Streamlit** cho dễ kiểm thử
- ✅ Báo cáo hiệu suất và tiêu thụ năng lượng chi tiết

---

## 🚀 Cài đặt

### Yêu cầu

- Python >= 3.8
- CUDA 11.8+ (nếu dùng GPU)
- pip hoặc conda

### Bước 1: Clone repo

```bash
git clone <your-repo-url>
cd ViSPhB
```

### Bước 2: Tạo môi trường ảo

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Hoặc cài theo từng mô-đun:

```bash
# Cho demo
pip install -r demo/app/requirements.txt

# Cho huấn luyện/đánh giá
pip install torch transformers datasets scikit-learn numpy pandas tqdm
```

---

## 📖 Hướng dẫn sử dụng

### 1️⃣ Chạy Demo Streamlit

```bash
streamlit run demo/app/app.py
```

Demo mở giao diện tương tác để:
- Chọn model (PhoBERT, Tier 1-3 với Direct/Two-stage)
- Nhập văn bản và nhận kết quả phân loại cảm xúc
- Xem confidence scores và efficiency metrics

**Chỉ định Model Root:**

```bash
VISPHB_MODEL_ROOT=models streamlit run demo/app/app.py
```

Hoặc thay đổi trực tiếp trong sidebar của ứng dụng.

### 2️⃣ Huấn luyện Mô hình

Các script huấn luyện nằm trong thư mục `code/`:

**Huấn luyện Two-Stage (Stage 1 Wiki KD → Tier 1/2/3):**

```bash
python code/tier1_spikebert_2stage.py --config <config-path> --seed 42
python code/tier2_implicit_2stage.py --config <config-path> --seed 42
python code/tier3_hybrid_2stage.py --config <config-path> --seed 42
```

**Huấn luyện Direct KD:**

```bash
python code/tier1_spikebert.py --config <config-path> --seed 42
python code/tier2_implicit.py --config <config-path> --seed 42
python code/tier3_hybrid.py --config <config-path> --seed 42
```

**Huấn luyện Baseline (PhoBERT):**

```bash
python code/phobert.py --config <config-path>
```

### 3️⃣ Nộp Job HPC

Script nộp công việc lên HPC cluster nằm trong `jobs/`:

```bash
# Two-stage
sbatch jobs/2stage/submit_tier1_spikebert_2stage.sh
sbatch jobs/2stage/submit_tier2_implicit_2stage.sh
sbatch jobs/2stage/submit_tier3_hybrid_2stage.sh

# Direct KD
sbatch jobs/direct/submit_tier1_spikebert.sh
sbatch jobs/direct/submit_tier2_implicit.sh
sbatch jobs/direct/submit_tier3_hybrid.sh

# Baseline
sbatch jobs/baseline/submit_phobert_baseline.sh
```

---

## 📁 Cấu trúc Project

```
ViSPhB/
├── code/                          # Script huấn luyện & inference
│   ├── phobert.py                # Baseline PhoBERT
│   ├── tier1_spikebert.py        # Tier 1 Direct
│   ├── tier1_spikebert_2stage.py # Tier 1 Two-stage
│   ├── tier2_implicit.py          # Tier 2 Direct
│   ├── tier2_implicit_2stage.py   # Tier 2 Two-stage
│   ├── tier3_hybrid.py            # Tier 3 Direct
│   ├── tier3_hybrid_2stage.py     # Tier 3 Two-stage
│   └── visualization_report.ipynb # Phân tích kết quả
│
├── data/                          # Dữ liệu
│   ├── uit-vsfc/                 # Dataset UIT-VSFC (phân loại cảm xúc)
│   │   ├── train/
│   │   ├── dev/
│   │   └── test/
│   └── wiki_vi_20231101_segmented/ # Wikipedia tiếng Việt (huấn luyện Stage 1)
│
├── models/                        # Mô hình đã huấn luyện
│   ├── baseline/
│   │   └── phobert_vsfc/         # PhoBERT VSFC baseline
│   ├── direct/                    # Direct KD (Tier 1/2/3)
│   │   ├── tier1_spikebert/
│   │   ├── tier2_implicit/
│   │   └── tier3_hybrid/
│   └── 2-stage/                   # Two-Stage KD (Tier 1/2/3)
│       ├── stage1_wiki_bptt/
│       ├── stage1_wiki_implicit/
│       ├── tier1_spikebert_2stage/
│       ├── tier2_implicit_2stage/
│       └── tier3_hybrid_2stage/
│
├── demo/
│   └── app/                       # Ứng dụng Streamlit
│       ├── app.py                # Điểm vào chính
│       ├── ui.py                 # Giao diện
│       ├── inference.py          # Inference logic
│       ├── registry.py           # Đăng ký mô hình
│       ├── reports.py            # Báo cáo
│       └── requirements.txt      # Dependencies
│
├── jobs/                          # Script nộp HPC
│   ├── baseline/
│   ├── direct/
│   └── 2stage/
│
├── paper-ref/                     # Tài liệu tham khảo
├── .gitignore                     # Git ignore
├── README.md                      # File này
└── prompt.txt                     # Hướng dẫn viết đồ án
```

---

## 📊 Các Mô hình & Kết quả

### Baseline

| Mô hình | Accuracy | F1-Score | Năng lượng (mJ) |
|---------|----------|----------|-----------------|
| PhoBERT | ~91% | ~0.90 | Baseline |

### Direct Knowledge Distillation

| Tier | Model | Accuracy | F1-Score | Năng lượng ↓ |
|------|-------|----------|----------|-------------|
| 1 | Tier 1 SpikeBERT | ~87% | ~0.86 | ~40% |
| 2 | Tier 2 Implicit | ~85% | ~0.84 | ~50% |
| 3 | Tier 3 Hybrid | ~83% | ~0.82 | ~55% |

### Two-Stage Knowledge Distillation

Huấn luyện Stage 1 trên Wikipedia → Tier 1/2/3 trên VSFC

| Tier | Stage 1 | Accuracy | F1-Score | Năng lượng ↓ |
|------|---------|----------|----------|-------------|
| 1 | Wiki BPTT | ~89% | ~0.88 | ~35% |
| 1 | Wiki Implicit | ~88% | ~0.87 | ~38% |
| 2 | Two-Stage | ~87% | ~0.86 | ~40% |
| 3 | Two-Stage | ~85% | ~0.84 | ~48% |

*Mức tiêu thụ năng lượng (Energy) tính theo so với PhoBERT baseline*

---

## 🔧 Cấu hình & Tuning

### Tạo config custom

```yaml
# config.yaml
model:
  type: tier1_spikebert
  num_tiers: 1
  threshold: 0.5

training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-5
  warmup_steps: 500
  seed: 42

knowledge_distillation:
  temperature: 4.0
  alpha: 0.7  # weight giữa KD loss và task loss

data:
  dataset: uit-vsfc
  train_split: train
  val_split: dev
  test_split: test
```

### Chạy với config tùy chỉnh

```bash
python code/tier1_spikebert.py --config config.yaml --seed 42
```

---

## 📈 Đánh giá & Phân tích

Xem notebook `code/visualization_report.ipynb` để:
- So sánh kết quả giữa các mô hình
- Phân tích biểu đồ hiệu suất & năng lượng
- Ablation study kết quả

```bash
jupyter notebook code/visualization_report.ipynb
```

---

## 🤝 Đóng góp

Nếu bạn tìm thấy lỗi hoặc có gợi ý cải tiến:

1. Fork repo
2. Tạo branch mới (`git checkout -b feature/improve-X`)
3. Commit thay đổi (`git commit -m 'Add feature X'`)
4. Push lên (`git push origin feature/improve-X`)
5. Tạo Pull Request

---

## 📝 Citation

Nếu bạn sử dụng ViSPhB trong nghiên cứu của mình, vui lòng trích dẫn:

```bibtex
@thesis{visphb2024,
  author = {Your Name},
  title = {Vietnamese Sentiment Classification with Spiking Neural Networks and Knowledge Distillation},
  school = {Your University},
  year = {2024}
}
```


---

## 🙏 Lời cảm ơn

- PhoBERT: [VinAI Research](https://github.com/VinAI/PhoBERT)
- UIT-VSFC Dataset: [University of Information Technology](https://github.com/uit-nlp/UIT-VSFC)
- PyTorch & Hugging Face Communities

