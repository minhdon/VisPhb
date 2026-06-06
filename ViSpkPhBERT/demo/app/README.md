# ViSPhB Streamlit Demo

Demo phân loại cảm xúc tiếng Việt cho PhoBERT teacher và các student SNN/KD của đồ án ViSPhB.

## Chạy nhanh

```bash
streamlit run demo/app/app.py
```

Có thể chọn model root bằng biến môi trường:

```bash
VISPHB_MODEL_ROOT=models streamlit run demo/app/app.py
VISPHB_MODEL_ROOT=uit-models streamlit run demo/app/app.py
```

Trong sidebar cũng có ô `Model root` để đổi nhanh giữa `models`, `uit-models` hoặc đường dẫn tuyệt đối.

## Cài package

```bash
pip install -r demo/app/requirements.txt
```

Các model SNN import lại class từ thư mục `code/`, nên môi trường cần có `torch`, `transformers`, `datasets`, `scikit-learn`, `numpy`, `pandas`.

## Cấu trúc thư mục model mong đợi

Layout hiện tại:

```text
models/
  baseline/
    phobert_vsfc/
      best_model.pth
      best_config.json
      test_results.json
      efficiency_report.json
  direct/
    tier1_spikebert/
      seed_42/
        best_tier1_seed42.pt
        energy_report_seed42.json
      seed_52/
      seed_62/
  2-stage/
    tier1_spikebert_2stage/
      spk2_T16/seed_42/
      spk4_T16/seed_42/
    tier2_implicit_2stage/
      spk2/seed_42/
      spk4/seed_42/
    tier3_hybrid_2stage/
      spk2/seed_42/
      spk4/seed_42/
```

Registry cũng thử layout legacy dạng:

```text
uit-models/
  phobert_vsfc/
  tier1_spikebert/seed_42/
  tier1_spikebert_2stage/spk2_T16/seed_42/
  tier2_implicit_2stage/spk2/seed_42/
  tier3_hybrid_2stage/spk2/seed_42/
```

## Checkpoint keys được hỗ trợ

App đọc các checkpoint `.pt` hoặc `.pth` có state dict nằm ở một trong các key:

- `model_state_dict`
- `student_state_dict`
- `state_dict`
- `model`

Nếu checkpoint là state dict trực tiếp, app cũng có thể nạp.

## Ghi chú vận hành

- `local_files_only` mặc định bật để tránh download online. Nếu máy chưa có cache HuggingFace cho `vinai/phobert-base-v2`, hãy đặt local model path vào ô `HuggingFace model/local path` hoặc tắt `local_files_only` trên máy có mạng.
- Nếu một model thiếu dependency, thiếu class architecture hoặc load checkpoint không khớp, app hiển thị lỗi tại UI và các model khác vẫn chạy được.
- Với model two-stage, registry chỉ khảo sát `spiking_layers=2` và `spiking_layers=4`.

