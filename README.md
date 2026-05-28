# Phân Tích Nghiên Cứu: SNN + Knowledge Distillation cho Vietnamese Sentiment Analysis

> **Tài liệu này phân tích 4 file training script Python cho đề tài nghiên cứu kết hợp Spiking Neural Networks (SNN) và Knowledge Distillation (KD) trên bài toán Vietnamese Sentiment Analysis với PhoBERT và UIT-VSFC.**

---

## A. Executive Summary (5–7 dòng)

Đề tài nghiên cứu huấn luyện **Spiking Neural Network student** từ **PhoBERT teacher đã fine-tuned** trên bài toán phân tích cảm xúc tiếng Việt (UIT-VSFC, 3 lớp: negative / neutral / positive). Toàn bộ 4 file thực hiện cùng một kiến trúc cốt lõi: PhoBERT backbone + classifier, trong đó các FFN layer ở top-k (k ∈ {3, 6, 9, 12}) được thay bằng LIF spiking layer. Attention layer **vẫn là ANN thuần túy**. Sự khác biệt chính giữa các file nằm ở (1) cơ chế gradient cho SNN — **SpikeBERTFFN** (T-step unrolling) vs **EquilibriumFFN** (equilibrium-state approximation), và (2) mức độ supervision từ teacher — chỉ logit KD, thêm feature alignment, hay thêm cả Stage 1 Wikipedia pre-alignment. Cả 4 file đều: dùng multi-seed (42/52/62), đo theoretical neuromorphic energy (MAC=4.6pJ / AC=0.9pJ), đo actual GPU energy qua pynvml, và lưu báo cáo JSON chi tiết. **Không có file tier3 riêng biệt** — logic tier3 được nhúng sẵn trong tất cả 4 file thông qua biến `EXPERIMENT_KIND`.

---

## B. Mapping Table

| File | `EXPERIMENT_KIND` | `TIER2_VARIANT` | Research Direction | Core Idea | Student Model | Stage 1 Wiki | Feature Align | Key Distinction |
|---|---|---|---|---|---|---|---|---|
| `tier1_spikingbert_impldiff.py` | `tier1` | `a` (placeholder) | **Tier 1 / SpikingBERT-style** | Equilibrium-state gradient approximation | `EquilibriumPhoBERTStudent` | ✅ Luôn chạy | ✅ Full-sequence MSE | Chạy LIF đến hội tụ ASR, gradient qua single-step unrolling, skip unconverged |
| `tier2_variant_a_logit_kd.py` | `tier2` | `a` | **Tier 2 – Variant A** | Logit-only KD | `SpikeBERTStudent` | ❌ | ❌ | Đơn giản nhất: chỉ CE + KL-div, không cần hidden states |
| `tier2_variant_b_feature_stage2.py` | `tier2` | `b` | **Tier 2 – Variant B** | Logit KD + Feature alignment ở Stage 2 | `SpikeBERTStudent` | ❌ | ✅ CLS-only MSE | Thêm MSE alignment trên CLS token, dùng learnable projection |
| `tier2_variant_c_full_twostage.py` | `tier2` | `c` | **Tier 2 – Variant C** | Full two-stage KD | `SpikeBERTStudent` | ✅ Chỉ khi variant=c | ✅ Full-sequence + Embedding | Stage 1 Wikipedia pre-alignment + Stage 2 full-loss với embedding alignment |

---

## C. Detailed Analysis Per Tier

---

### 🔵 Tier 1: SpikingBERT-style / Equilibrium-state Gradient Approximation

**File:** [`tier1_spikingbert_impldiff.py`](./tier1_spikingbert_impldiff.py)

#### Mục tiêu nghiên cứu
Triển khai phiên bản gần đúng của "implicit differentiation" theo tinh thần SpikingBERT (Bal & Sengupta 2023). Thay vì unroll T bước cố định (SpikeBERT), forward pass chạy LIF đến khi **Average Spike Rate (ASR) hội tụ** (δ < ε), gradient được tính qua **single-step Arctan surrogate** tại điểm cân bằng — bỏ qua toàn bộ computation graph của T bước hội tụ.

#### Teacher & Student
- **Teacher:** `PhoBERTTeacher` — PhoBERT backbone + linear classifier, load từ checkpoint `phobert_vsfc/best_model.pth`, freeze toàn bộ (`param.requires_grad_(False)`)
- **Student:** `EquilibriumPhoBERTStudent` — PhoBERT backbone, top-k FFN layers thay bằng `EquilibriumFFN`

#### Key code evidence

```python
# tier1_spikingbert_impldiff.py, line 72-73
EXPERIMENT_KIND = "tier1"

# build_model() — line 193-194: chọn EquilibriumPhoBERTStudent
if EXPERIMENT_KIND in {"tier1", "tier3"}:
    student = EquilibriumPhoBERTStudent(args, teacher_ckpt)

# EquilibriumFFN.forward() — lines 604-620: chạy LIF đến equilibrium
asr_star, record = run_lif_to_equilibrium(
    current.detach(), self.t_conv, self.threshold, self.gamma, self.eps
)
# Straight-through trick tại điểm cân bằng:
spike_grad = ArctanSpike.apply(mem_at_eq - self.threshold, k)
asr_with_grad = asr_star.detach() + (spike_grad - spike_grad.detach())

# should_run_stage1() — line 1448-1449: Tier 1 LUÔN chạy Stage 1
if EXPERIMENT_KIND == "tier1":
    return not args.skip_stage1
```

#### Đặc điểm phân biệt
| Thuộc tính | Giá trị |
|---|---|
| KD | ✅ (CE + KL-div + feature alignment) |
| Feature alignment | ✅ Full-sequence MSE tại layers 3,6,9,12 |
| Embedding alignment | ✅ |
| Stage 1 Wikipedia | ✅ |
| Spiking FFN | `EquilibriumFFN` (chạy đến hội tụ, up to `t_conv=80` steps) |
| Skip unconverged | ✅ `has_unconverged()` — bỏ batch nếu ASR chưa hội tụ |
| Đo energy | ✅ Theoretical + GPU |
| Multi-seed | ✅ [42, 52, 62] |

---

### 🟢 Tier 2 – Variant A: Logit KD Only

**File:** [`tier2_variant_a_logit_kd.py`](./tier2_variant_a_logit_kd.py)

#### Mục tiêu nghiên cứu
Baseline SpikeBERT-style KD đơn giản nhất: chỉ dùng **KL-divergence trên logits** (soft targets từ teacher). Không cần truyền hidden states, không có Stage 1. Đây là điểm xuất phát để ablation: thêm feature alignment có cải thiện không?

#### Teacher & Student
- **Teacher:** `PhoBERTTeacher` — PhoBERT đã fine-tuned, freeze
- **Student:** `SpikeBERTStudent` — PhoBERT backbone + top-k FFN thay bằng `SpikeBERTFFN` (T-step fixed unrolling, T=4 mặc định)

#### Key code evidence

```python
# tier2_variant_a_logit_kd.py, lines 72-73
EXPERIMENT_KIND = "tier2"
TIER2_VARIANT = "a"

# compute_stage2_loss() — line 1322-1323: chỉ CE + KD
if EXPERIMENT_KIND == "tier2" and TIER2_VARIANT == "a":
    loss = args.alpha_kd * ce + (1.0 - args.alpha_kd) * kd

# needs_feature_alignment() — line 1312-1313: Variant A KHÔNG cần hidden states
def needs_feature_alignment() -> bool:
    return TIER2_VARIANT in {"b", "c"} or EXPERIMENT_KIND in {"tier1", "tier3"}
# → False cho tier2-a, nên teacher/student KHÔNG trả hidden_states

# SpikeBERTFFN.forward() — lines 491-501: unroll T steps cố định
for _ in range(self.t_steps):          # t_steps=4
    mem = current + beta * mem
    spike = ArctanSpike.apply(mem - self.threshold, self.k_slope)
    mem = mem - spike * self.threshold
    spike_sum = spike_sum + spike
    out_sum = out_sum + self.output_dense(spike)
return out_sum / float(self.t_steps)

# should_run_stage1() — line 1446-1447: KHÔNG chạy Stage 1 cho variant a
if EXPERIMENT_KIND == "tier2":
    return TIER2_VARIANT == "c" and not args.skip_stage1
```

#### Đặc điểm phân biệt
| Thuộc tính | Giá trị |
|---|---|
| KD | ✅ CE + KL-div |
| Feature alignment | ❌ |
| Embedding alignment | ❌ |
| Stage 1 Wikipedia | ❌ |
| Spiking FFN | `SpikeBERTFFN` (T-step fixed, β learned per-neuron) |
| Skip unconverged | ❌ (không dùng EquilibriumFFN) |
| Đo energy | ✅ |
| Multi-seed | ✅ |

---

### 🟡 Tier 2 – Variant B: Logit KD + Feature Alignment ở Stage 2

**File:** [`tier2_variant_b_feature_stage2.py`](./tier2_variant_b_feature_stage2.py)

#### Mục tiêu nghiên cứu
Mở rộng Variant A bằng cách thêm **feature alignment loss** trực tiếp ở Stage 2 (fine-tuning trên UIT-VSFC). Alignment chỉ trên **CLS token** (`cls_only=True`), qua learnable linear projection. Câu hỏi: feature alignment có giúp SNN student học representation tốt hơn mà không cần Stage 1 không?

#### Key code evidence

```python
# tier2_variant_b_feature_stage2.py, lines 72-73
EXPERIMENT_KIND = "tier2"
TIER2_VARIANT = "b"

# compute_stage2_loss() — lines 1324-1331
elif EXPERIMENT_KIND == "tier2" and TIER2_VARIANT == "b":
    feature = feature_alignment_loss(
        student_out["hidden_states"],
        teacher_out["hidden_states"],
        cls_only=True,          # ← chỉ CLS token, khác Variant C
        student_model=student_model,
    )
    loss = args.alpha_kd * ce + (1.0 - args.alpha_kd) * kd + args.feature_weight * feature

# feature_alignment_loss() — lines 1284-1305: learnable projection
projections = getattr(student_model, "feature_projections", {})
for idx in layers:   # layers [3, 6, 9, 12]
    s_val = student_hidden[idx].float()
    t_val = teacher_hidden[idx].detach().float()
    if str(idx) in projections:
        s_val = projections[str(idx)](s_val)  # Linear + LayerNorm
    if cls_only:
        s_val = s_val[:, 0, :]   # chỉ CLS
        t_val = t_val[:, 0, :]
    losses.append(F.mse_loss(s_val, t_val))
```

#### Đặc điểm phân biệt so với Variant A và C
| Thuộc tính | Variant A | Variant B | Variant C |
|---|---|---|---|
| Feature alignment | ❌ | ✅ CLS-only | ✅ Full-sequence |
| Embedding alignment | ❌ | ❌ | ✅ |
| Stage 1 Wikipedia | ❌ | ❌ | ✅ |
| Loss formula | α·CE + (1-α)·KD | α·CE + (1-α)·KD + λ·Feat | 0.1·Feat + 0.1·Emb + KD + 0.1·CE |

---

### 🔴 Tier 2 – Variant C: Full Two-Stage KD

**File:** [`tier2_variant_c_full_twostage.py`](./tier2_variant_c_full_twostage.py)

#### Mục tiêu nghiên cứu
Triển khai **đầy đủ** quy trình KD hai giai đoạn theo tinh thần SpikeBERT gốc:
- **Stage 1:** Unsupervised hidden-state alignment trên Vietnamese Wikipedia (wikimedia/wikipedia 20231101.vi) — không cần nhãn
- **Stage 2:** Fine-tuning có giám sát trên UIT-VSFC với full loss (feature + embedding + KD + CE)

Câu hỏi nghiên cứu: Stage 1 Wikipedia có cần thiết khi student đã được khởi tạo từ PhoBERT (vốn được pre-train trên tiếng Việt)?

#### Key code evidence

```python
# tier2_variant_c_full_twostage.py, line 73
TIER2_VARIANT = "c"

# should_run_stage1() — line 1446-1447: CHỈ variant C chạy Stage 1
if EXPERIMENT_KIND == "tier2":
    return TIER2_VARIANT == "c" and not args.skip_stage1

# train_stage1_epoch() — lines 1411-1417: Stage 1 loss = feature + embedding alignment
loss = feature_alignment_loss(
    student_out["hidden_states"],
    teacher_out["hidden_states"],
    cls_only=False,          # full sequence
    student_model=student,
)
loss = loss + embedding_alignment_loss(student_out["hidden_states"], teacher_out["hidden_states"])

# load_wiki_loader() — line 1214: dataset Wikipedia tiếng Việt
ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", ...)

# compute_stage2_loss() variant C — lines 1332-1340: Stage 2 full loss
elif EXPERIMENT_KIND == "tier2" and TIER2_VARIANT == "c":
    feature = feature_alignment_loss(..., cls_only=False)
    embedding = embedding_alignment_loss(...)
    loss = 0.1 * feature + 0.1 * embedding + kd + 0.1 * ce
```

---

### 🟣 Tier 3 / Hybrid (Nhúng trong tất cả 4 file)

> **Lưu ý:** Không có file `tier3_*.py` riêng. Logic tier3 được nhúng trong tất cả 4 file thông qua `EXPERIMENT_KIND = "tier3"` (hardcoded trong tier1 file) và `hybrid_variant` argument.

**Hai submode:**
- **3a** (SpikeBERT-style): Loss = CE + KD (không feature alignment, không Stage 1)
- **3b** (Full hybrid): Loss = 0.1·Feat + 0.1·Emb + KD + 0.1·CE + Stage 1 Wikipedia

```python
# compute_stage2_loss() — line 1341-1351
elif EXPERIMENT_KIND == "tier3" and getattr(args, "_active_submode", "3b") == "3a":
    loss = ce + kd
else:  # tier3-3b
    feature = feature_alignment_loss(..., cls_only=False)
    embedding = embedding_alignment_loss(...)
    loss = 0.1 * feature + 0.1 * embedding + kd + 0.1 * ce

# should_run_stage1() — line 1450-1451
if EXPERIMENT_KIND == "tier3":
    return submode == "3b" and not args.skip_stage1  # CHỈ 3b có Stage 1

# Student trong tier3 = EquilibriumPhoBERTStudent (giống Tier 1)
# → Hybrid: EquilibriumFFN (gradient) + full KD loss (như Variant C)
```

---

## D. Code Evidence Table

| Claim | Code Evidence | Dòng | Giải thích cho slide |
|---|---|---|---|
| PhoBERT là backbone | `MODEL_NAME = "vinai/phobert-base-v2"` + `AutoModel.from_pretrained(MODEL_NAME)` | L76, L990-997 | PhoBERT v2 là transformer tiếng Việt từ VinAI |
| Teacher là PhoBERT fine-tuned | `PhoBERTTeacher.__init__` load `best_model.pth` | L436-445 | Checkpoint từ fine-tuning riêng trước đó |
| Teacher bị freeze hoàn toàn | `for param in teacher.parameters(): param.requires_grad_(False)` | L190-191 | Teacher chỉ cung cấp soft targets, không cập nhật |
| Student init từ PhoBERT | `SpikeBERTStudent.__init__` / `EquilibriumPhoBERTStudent.__init__` gọi `load_backbone(args)` rồi load state dict của teacher checkpoint | L514-537, L632-657 | Warm-start từ pretrained weights |
| Attention vẫn là ANN | `layer_module.attention(hidden_states, ...)` — dùng nguyên layer gốc | L552-558, L673-679 | Chỉ FFN bị thay, attention giữ nguyên |
| FFN replaced by spiking layer | `self.spiking_ffns[str(idx)] = SpikeBERTFFN(...)` | L530-532 | Chỉ top-k layer được thay |
| `spiking_layers` ∈ {3,6,9,12} | `parser.add_argument("--spiking-layers", choices=[3, 6, 9, 12])` | L131 | Ablation: thay 3 / 6 / 9 / 12 layer cuối |
| LIF neuron implement | `SpikeBERTFFN.forward()` — vòng lặp T steps, mem, spike | L485-501 | LIF với membrane potential tích lũy |
| Arctan surrogate gradient | `class ArctanSpike(Function)` với `backward` dùng `alpha/(2*(1+(alpha*x)²))` | L458-469 | Surrogate gradient để backprop qua spike |
| `beta` (membrane decay) | `self.beta = nn.Parameter(torch.full(..., float(args.beta)))` | L480 | β là learnable per-neuron |
| Firing rate tracking | `self.total_spikes / self.total_neurons` | L510 | Dùng cho energy estimate |
| EquilibriumFFN khác SpikeBERTFFN | `run_lif_to_equilibrium()` — chạy đến δ < ε thay vì T cố định; gradient qua single-step tại ASR* | L1355-1372, L615-618 | Tránh lưu T-step graph, gradient tốt hơn straight-through |
| CE loss | `F.cross_entropy(student_out["logits"].float(), labels)` | L1317 | Standard supervised loss |
| KL-div KD loss | `F.kl_div(F.log_softmax(student/T), F.softmax(teacher/T)) * T²` | L1272-1277 | Hinton KD với temperature scaling |
| Feature alignment loss | `F.mse_loss(s_val, t_val)` tại layers [3,6,9,12] qua projection | L1284-1305 | Align hidden states giữa teacher/student |
| Embedding alignment loss | `F.mse_loss(student_hidden[0], teacher_hidden[0])` | L1308-1309 | Align embedding layer (layer 0) |
| Stage 1 loss | `feature_alignment_loss + embedding_alignment_loss` trên Wikipedia | L1411-1417 | Không có CE vì không có nhãn |
| Stage 2 loss (Variant B) | `α·CE + (1-α)·KD + λ·Feature_CLS` | L1331 | Thêm alignment trực tiếp vào fine-tuning |
| Stage 2 loss (Variant C) | `0.1·Feat + 0.1·Emb + KD + 0.1·CE` | L1340 | Tỷ trọng khác Variant B |
| Teacher freeze (evidence) | `teacher.eval()` + `requires_grad_(False)` + `torch.no_grad()` khi forward teacher | L189-191, L246 | Teacher không nhận gradient |
| Teacher logits | `teacher_out["logits"]` từ `self.classifier(cls)` của teacher | L455, L1318 | Dùng làm soft targets |
| Student logits | `student_out["logits"]` từ forward pass của student | L571, L693 | Đầu ra của student classifier |
| KD chứ không phải fine-tuning | Teacher logits dùng làm soft targets trong KL-div; teacher không update; student học mimic teacher | L246-247, L1317-1323 | Nếu chỉ fine-tuning thì không cần teacher |
| Theoretical energy estimate | `estimate_energy()`: MAC=4.6pJ cho ANN, AC=0.9pJ×rate cho SNN | L1484-1553 | Dựa Horowitz 2014, 45nm CMOS |
| ANN MAC vs SNN AC | `sops = t_steps * rate * ffn_flops; ffn_energy = sops * AC_PJ` | L1516-1518 | Energy tỷ lệ với firing rate |
| Actual GPU energy via pynvml | `_pynvml.nvmlDeviceGetPowerUsage(handle)` trong inference loop | L1592 | Đo watt thực tế, tính J = W × s |
| Multi-seed | `--seeds [42, 52, 62]` + vòng lặp `for seed in args.seeds` | L115, L735 | Báo cáo mean ± std |

---

## E. Slide Outline

---

### Slide 1: Problem & Motivation

**Tiêu đề:** Hướng tới AI tiếng Việt tiết kiệm năng lượng — SNN cho Sentiment Analysis

**Bullets:**
- Phân tích cảm xúc tiếng Việt (Vietnamese Sentiment Analysis) là bài toán NLP cơ bản nhưng quan trọng cho hệ thống review, mạng xã hội
- Các mô hình BERT hiện đại (PhoBERT) đạt hiệu năng cao nhưng **tốn kém về mặt tính toán và năng lượng**
- Spiking Neural Networks (SNN) hoạt động theo kiểu **event-driven** — chỉ tốn năng lượng khi có spike, lý tưởng cho hardware neuromorphic
- Thách thức: SNN khó huấn luyện trực tiếp trên NLP task → cần **Knowledge Distillation** từ ANN teacher mạnh

**Code evidence:** `MODEL_NAME = "vinai/phobert-base-v2"` (L76); `NUM_LABELS = 3` (L77); `MAC_PJ = 4.6; AC_PJ = 0.9` (L79-80)

**Visual suggestion:** So sánh sơ đồ ANN vs SNN — bên trái: dense multiply-accumulate; bên phải: sparse spike-based accumulate

**Speaker notes:** "Vấn đề không chỉ là accuracy — mà là accuracy-per-joule. SNN lý thuyết có thể tiết kiệm năng lượng gấp nhiều lần so với ANN khi firing rate thấp."

---

### Slide 2: Research Gap

**Tiêu đề:** SNN + BERT: Những gì còn chưa được giải quyết

**Bullets:**
- SpikeBERT (Zhu et al. 2023): dùng fixed T-step LIF + KD từ BERT → **chưa thử trên tiếng Việt, chưa có feature alignment**
- SpikingBERT (Bal & Sengupta 2023): equilibrium-state gradient → **chưa áp dụng trong KD pipeline đầy đủ**
- Chưa có nghiên cứu so sánh hệ thống: logit KD vs feature alignment vs two-stage Wiki pretraining trên **Vietnamese NLP**
- Câu hỏi mở: **PhoBERT initialization có làm Stage 1 Wikipedia trở nên dư thừa không?**

**Code evidence:** Comment trong `EquilibriumFFN`: *"approximation of implicit differentiation (SpikingBERT, Bal & Sengupta 2023)"* (L583-588)

**Visual suggestion:** Bảng so sánh SpikeBERT / SpikingBERT / công trình này — các cột: Language, Gradient Method, KD Type, Stage 1

**Speaker notes:** "Chúng tôi không chỉ tái hiện SpikeBERT — chúng tôi so sánh hệ thống 4 hướng, và lần đầu thực nghiệm trên tiếng Việt với PhoBERT."

---

### Slide 3: Overall Framework

**Tiêu đề:** Framework tổng quan: Teacher → Student với SNN FFN

**Bullets:**
- **Teacher:** PhoBERT-base-v2 fine-tuned trên UIT-VSFC (ANN, freeze trong training)
- **Student:** PhoBERT backbone (warm-start từ teacher weights) với top-k FFN thay bằng SNN layer
- **Distillation:** Teacher cung cấp soft logits + hidden states; student học mimick qua nhiều loss
- **Evaluation:** Accuracy, F1-weighted, F1-macro trên 3 seeds → báo cáo mean ± std

**Code evidence:** `build_model()` (L181-205): teacher freeze, student init từ teacher checkpoint; `evaluate()` (L288-344): metrics + firing rates

**Visual suggestion:** Sơ đồ pipeline: [UIT-VSFC] → [PhoBERT Teacher] ⟶KD⟶ [SNN Student] → [Classification Head] → [3-class output]

**Speaker notes:** "Student được warm-start từ chính teacher — đây là quyết định quan trọng ảnh hưởng đến câu hỏi Stage 1 có cần không."

---

### Slide 4: Teacher-Student Architecture

**Tiêu đề:** Kiến trúc Hybrid: ANN Attention + SNN FFN

**Bullets:**
- Kiến trúc PhoBERT: 12 Transformer layers, mỗi layer = Multi-Head Attention + FFN
- **Attention layers: giữ nguyên ANN** (dense MACs, không thay đổi)
- **Top-k FFN layers (k ∈ {3,6,9,12}): thay bằng LIF spiking FFN** — spike binary, accumulate thay vì multiply
- Learnable projection `Linear + LayerNorm` tại mỗi checkpoint layer [3,6,9,12] để alignment
- Teacher được freeze hoàn toàn: chỉ forward pass, không backprop

**Code evidence:**
```python
# Attention: dùng nguyên layer_module.attention()
attention_outputs = layer_module.attention(hidden_states, ...)  # L552

# FFN: thay bằng spiking
ffn_output = self.spiking_ffns[str(idx)](attention_output)     # L559

# Teacher freeze:
for param in teacher.parameters():
    param.requires_grad_(False)                                 # L191
```

**Visual suggestion:** Diagram: 12-layer stack, layers 1-9 màu xanh (ANN), layers 10-12 màu đỏ (SNN); mũi tên đứt đoạt cho spike

**Speaker notes:** "Attention không được thay vì spike-based attention cho text có nhiều vấn đề về gradient. Đây là hybrid approach — giữ expressiveness của attention, tiết kiệm năng lượng ở FFN."

---

### Slide 5: Tier 1 / Hướng 1 — SpikingBERT-style Equilibrium Gradient

**Tiêu đề:** Tier 1: Equilibrium-state Gradient Approximation

**Bullets:**
- **Forward:** Chạy LIF simulation đến khi ASR hội tụ (δ < ε, tối đa `t_conv=80` steps) — **không lưu computation graph**
- **Backward:** Tính gradient tại điểm cân bằng ASR* bằng Arctan surrogate — single-step unrolling
- Nếu ASR không hội tụ: **bỏ qua batch** (`skipped_steps`) — tránh gradient nhiễu
- Luôn chạy Stage 1 Wikipedia + Stage 2 với full loss (Feat + Emb + KD + CE)
- Lý do chọn equilibrium: tránh lưu T-step graph (memory), gradient tốt hơn straight-through

**Code evidence:**
```python
# EquilibriumFFN.forward(): run đến equilibrium, gradient qua single-step
asr_star, record = run_lif_to_equilibrium(
    current.detach(), self.t_conv, ...)  # L607-608
spike_grad = ArctanSpike.apply(mem_at_eq - self.threshold, k)
asr_with_grad = asr_star.detach() + (spike_grad - spike_grad.detach())  # L618

# Skip unconverged:
if EXPERIMENT_KIND in {"tier1"} and has_unconverged(student_out.get("convergence", [])):
    skipped_steps += 1; continue  # L251-253
```

**Visual suggestion:** Biểu đồ: trục x = timestep, trục y = ASR; đường converge tại t* → điểm đánh dấu là nơi tính gradient

**Speaker notes:** "Khác biệt cốt lõi với SpikeBERT: thay vì T=4 bước cố định, chúng tôi chờ hội tụ. Điều này gần với 'implicit differentiation' hơn nhưng vẫn tractable."

---

### Slide 6: Tier 2 — Ba Variant của SpikeBERT KD

**Tiêu đề:** Tier 2: Hệ thống Ablation trên SpikeBERT Student

**Bullets:**
- Cùng student model: `SpikeBERTStudent` (T-step fixed, β learned per-neuron)
- **Variant A (Baseline):** `L = α·CE + (1-α)·KD` — chỉ soft logits
- **Variant B (+ Feature Stage 2):** `L = α·CE + (1-α)·KD + λ·MSE(CLS_student, CLS_teacher)` — thêm CLS alignment
- **Variant C (Full Two-Stage):** Stage 1 Wiki `L=Feat+Emb`, Stage 2 `L=0.1·Feat + 0.1·Emb + KD + 0.1·CE` — full pipeline

**Code evidence:**
```python
# compute_stage2_loss() — lines 1322-1340
if   TIER2_VARIANT == "a": loss = alpha*ce + (1-alpha)*kd
elif TIER2_VARIANT == "b": loss = alpha*ce + (1-alpha)*kd + lw*feature  # cls_only=True
elif TIER2_VARIANT == "c": loss = 0.1*feature + 0.1*embedding + kd + 0.1*ce  # full seq
```

**Visual suggestion:** Bảng 3 cột A/B/C với checkmarks cho từng loss component; mũi tên progression từ A→B→C

**Speaker notes:** "A, B, C là ablation có hệ thống. Mục tiêu: biết chính xác contribution của feature alignment và Stage 1."

---

### Slide 7: Tier 3 — Hybrid: Equilibrium Gradient + Full KD

**Tiêu đề:** Tier 3 Hybrid: Kết hợp tốt nhất của Tier 1 và Tier 2C

**Bullets:**
- **Student:** `EquilibriumPhoBERTStudent` (gradient equilibrium như Tier 1)
- **KD:** Full loss như Tier 2 Variant C (feature + embedding + logit)
- **Submode 3a:** CE + KD (baseline hybrid, không Stage 1)
- **Submode 3b:** Full Stage 1 Wikipedia + Stage 2 full loss
- So sánh 3a vs 3b trong cùng một lần chạy → kiểm chứng Stage 1 có cần không với equilibrium student

**Code evidence:**
```python
# submodes cho tier3 — main() line 725-726
if EXPERIMENT_KIND == "tier3":
    submodes = ["3a","3b"] if args.hybrid_variant=="both" else [args.hybrid_variant]

# compute_stage2_loss() — line 1341-1351
elif EXPERIMENT_KIND == "tier3" and submode == "3a":
    loss = ce + kd
else:  # 3b
    loss = 0.1*feature + 0.1*embedding + kd + 0.1*ce
```

**⚠️ Lưu ý:** Không có file `tier3_*.py` riêng biệt trong thư mục. Logic tier3 được hard-code trong `tier1_spikingbert_impldiff.py` (dòng 72: `EXPERIMENT_KIND = "tier1"`). Để chạy tier3, cần thay `EXPERIMENT_KIND = "tier3"` trong code.

**Visual suggestion:** Venn diagram: Tier1 ∩ Tier2C = Tier3; hoặc bảng 2×2: {Equilibrium, Fixed-T} × {No-Stage1, Stage1}

**Speaker notes:** "Tier 3 không phải một file riêng — nó là một configuration mode nhúng trong codebase. Đây là điểm cần clarify khi present."

---

### Slide 8: Training Pipeline

**Tiêu đề:** Training Pipeline: Multi-Seed với Auto-Resume

**Bullets:**
- **Multi-seed loop:** seeds = [42, 52, 62] → báo cáo mean ± std (reproducibility)
- **Stage 1 (nếu applicable):** Wikipedia → học align hidden states không giám sát
- **Stage 2:** UIT-VSFC → supervised KD với loss tương ứng từng variant
- **Checkpointing:** epoch-level checkpoint + best-dev-F1 checkpoint + auto-resume
- **Output:** JSON report với test metrics + energy (theoretical + GPU) + firing rates

**Code evidence:**
```python
for seed in args.seeds:              # L735 — multi-seed loop
    if should_run_stage1(args, submode):
        stage1_history = run_stage1(bundle, tokenizer, device)  # L769
    # Stage 2 training loop
    for epoch in range(start_epoch, args.epochs + 1):           # L796
        train_stats = train_one_epoch(...)
        dev_metrics, _, _ = evaluate(...)
    # Energy reporting
    theoretical_energy = estimate_energy(bundle.student, args)  # L869
    gpu_energy = measure_gpu_energy(bundle, test_loader, ...)   # L870
```

**Visual suggestion:** Flowchart: START → [Stage 1?] → Stage 2 epochs → Eval → Save → Next seed → Summarize

**Speaker notes:** "Auto-resume quan trọng vì training có thể mất nhiều giờ. Nếu GPU bị preempt, script tự tiếp tục từ epoch cuối."

---

### Slide 9: Ablation Design

**Tiêu đề:** Ablation Design: Từng Yếu Tố Được Kiểm Soát

**Bullets:**
- **Ablation 1 — Gradient method:** EquilibriumFFN (Tier 1) vs SpikeBERTFFN (Tier 2) — giữ nguyên loss
- **Ablation 2 — KD richness:** Variant A (logit) < B (logit+CLS) < C (logit+full-seq+emb)
- **Ablation 3 — Stage 1:** Variant B vs C, Tier 3 submode 3a vs 3b
- **Ablation 4 — Spiking depth:** `--spiking-layers` ∈ {3, 6, 9, 12} — bao nhiêu layer SNN là tối ưu?
- Tất cả đều multi-seed → kết quả thống kê tin cậy

**Code evidence:** `--spiking-layers choices=[3,6,9,12]` (L131); `top_layer_indices(num_layers, spiking_layers)` (L1228-1230); `should_run_stage1()` (L1445-1452)

**Visual suggestion:** Ma trận ablation: hàng = Gradient method, cột = KD type; ô = file/variant tương ứng

**Speaker notes:** "Thiết kế ablation cho phép tách biệt contribution của từng thành phần. Đây là điểm mạnh của nghiên cứu."

---

### Slide 10: Energy Measurement

**Tiêu đề:** Đo năng lượng: Theoretical vs Actual

**Bullets:**
- **Theoretical (neuromorphic, 45nm):** ANN layers = MACs × 4.6pJ; SNN FFN = T × rate × FLOPs × 0.9pJ
- **Actual GPU:** pynvml đo watt thực tế trong inference → Joule/sample
- **Energy reduction %** = `(1 - total_energy / total_ann_energy) × 100%`
- SNN tiết kiệm khi **firing rate thấp** (sparse spikes) — đây là điều cần chứng minh thực nghiệm
- GPU energy của SNN cao hơn ANN vì T-step simulation trên GPU không được tối ưu (expected!)

**Code evidence:**
```python
# estimate_energy() — lines 1516-1522
if is_spiking:
    sops = t_steps * rate * ffn_flops   # rate từ firing_rate tracker
    ffn_energy = sops * AC_PJ           # 0.9 pJ mỗi AC op
else:
    ffn_energy = ffn_flops * MAC_PJ     # 4.6 pJ mỗi MAC op

# measure_gpu_energy() — lines 1592, 1608-1609
power_readings_mw.append(float(_pynvml.nvmlDeviceGetPowerUsage(handle)))
avg_power_w = sum(power_readings_mw) / len(power_readings_mw) / 1000.0
total_energy_j = avg_power_w * elapsed
```

**Visual suggestion:** Bar chart đôi: trục x = {Tier1, Tier2A, Tier2B, Tier2C, Teacher}; màu xanh = theoretical, màu đỏ = GPU actual; chú thích firing rate

**Speaker notes:** "Hai loại energy phản ánh hai câu chuyện khác nhau: theoretical cho thấy tiềm năng neuromorphic hardware; GPU actual phản ánh thực tế GPU training."

---

### Slide 11: Expected Results / What to Compare

**Tiêu đề:** Kế hoạch So sánh Kết quả (Planned Evaluation)

**Bullets (PLANNED — chưa có số thật):**
- Baseline: Teacher PhoBERT F1-weighted trên UIT-VSFC (expected high)
- So sánh chính: Tier2A < Tier2B ≤ Tier2C ≤ Tier1 (về F1) — kiểm chứng thực nghiệm
- So sánh Stage 1: Tier2B vs Tier2C — Stage 1 Wiki có giúp với PhoBERT init không?
- So sánh gradient: Tier1 vs Tier2C — EquilibriumFFN có tốt hơn SpikeBERTFFN không?
- Energy: Tier1 theoretical energy reduction (%) khi firing_rate < 0.2

**⚠️ Không có số liệu thực tế trong code — chưa chạy xong.**

**Code evidence:** `summarize()` — tính mean/std qua seeds (L1668-1675); `seed_report` chứa `test_metrics`, `energy` (L871-884)

**Visual suggestion:** Placeholder table với "TBD" — hoặc kẻ sẵn bảng 5 hàng × 4 cột để điền sau

**Speaker notes:** "Chúng tôi không có số liệu chính thức tại thời điểm present. Nhưng framework ablation đã rõ ràng — câu hỏi nghiên cứu được kiểm chứng bằng code thực."

---

### Slide 12: Risks & Limitations

**Tiêu đề:** Rủi ro và Giới hạn

**Bullets:**
- **EquilibriumFFN convergence:** Nếu ASR không hội tụ, batch bị skip → có thể mất nhiều data trong training
- **Stage 1 thực sự cần hay không?** — PhoBERT init đã mạnh, Stage 1 Wiki có thể không giúp nhiều
- **GPU energy ≠ neuromorphic energy:** SNN chạy trên GPU thực ra tốn hơn ANN — phải nói rõ đây là *theoretical* estimate
- **UIT-VSFC nhỏ** (~16,000 mẫu train) → SNN student dễ overfit nếu feature alignment quá mạnh
- **Tier 3 chưa có file riêng** — không thể chạy tier3 trực tiếp mà không sửa code

**Code evidence:**
- `skipped_steps` tracking (L252-253, L282): ghi lại số batch bị bỏ
- Comment trong `measure_gpu_energy()` (L1611-1614): *"Higher than teacher is expected for T-step SNN simulation on GPU"*
- `has_unconverged()` (L1375-1376)

**Visual suggestion:** Risk matrix: trục x = xác suất, trục y = impact; 4 rủi ro trên được đặt vào

**Speaker notes:** "Quan trọng phải nói thẳng: GPU energy của SNN student có thể cao hơn teacher. Lợi ích về energy chỉ thực sự phát huy trên neuromorphic hardware."

---

### Slide 13: Key Contributions

**Tiêu đề:** Đóng góp Nghiên cứu

**Bullets:**
- **[C1] Lần đầu áp dụng SNN + KD trên Vietnamese Sentiment Analysis** (PhoBERT + UIT-VSFC)
- **[C2] Hệ thống ablation đầy đủ:** So sánh 4 hướng (gradient method × KD richness)
- **[C3] Triển khai equilibrium-state gradient** cho SNN trong BERT pipeline (approximation của SpikingBERT)
- **[C4] Phân tích năng lượng dual:** Theoretical neuromorphic energy + actual GPU energy
- **[C5] Câu hỏi nghiên cứu mới:** PhoBERT initialization có làm Stage 1 Wikipedia dư thừa không?

**Code evidence:** Toàn bộ framework: `EXPERIMENT_KIND` switch, `estimate_energy()`, `measure_gpu_energy()`, `run_lif_to_equilibrium()`

**Visual suggestion:** Timeline/roadmap: Teacher training → [4 experiments] → Ablation comparison → Energy analysis

**Speaker notes:** "C5 là câu hỏi thú vị nhất — nếu PhoBERT init đã đủ mạnh, thì Stage 1 Wiki tốn compute mà không thêm ích lợi — điều này có ý nghĩa thực tiễn cho resource-limited settings."

---

## F. Critical Notes (Cẩn thận khi trình bày)

### ⚠️ Những điểm KHÔNG được claim quá mạnh:

1. **Không có số liệu accuracy/F1 thực tế** — code chưa chạy xong. Tuyệt đối không tự đặt số.

2. **"Equilibrium gradient" ≠ full implicit differentiation** — code comment (L583-588) nói rõ: *"It is not full implicit differentiation via Anderson acceleration, but it is tractable for GPU training."* Chỉ nói là "approximation".

3. **GPU energy > Teacher energy** — code comment (L1611-1614) thừa nhận điều này. SNN trên GPU không tiết kiệm hơn ANN trên GPU. Chỉ tiết kiệm trên neuromorphic hardware (theoretical).

4. **Tier 3 không có file riêng** — logic tier3 embed trong 4 file. Nếu ai hỏi "file tier3 đâu?" cần giải thích.

5. **`beta` trong SpikeBERTFFN là per-neuron learnable** — không phải hyperparameter cố định. Đây là điểm cần giải thích cho audience.

6. **Feature alignment ở Tier 1 và Tier 2 khác nhau:**
   - Tier 1: `cls_only=False` (full sequence)
   - Tier 2 Variant B: `cls_only=True` (chỉ CLS)
   - Tier 2 Variant C: `cls_only=False` (full sequence)

7. **Không có embedding alignment trong Tier 2B** — chỉ có trong Variant C và Tier 1/3.

8. **Tier 1 LUÔN chạy Stage 1** (không thể bỏ trừ khi `--skip-stage1`) → thời gian training dài hơn nhiều.

---

## G. Minimum Slide Version (7 phút)

Nếu chỉ có **7 phút**, giữ các slide sau và rút gọn:

| # | Slide | Thời gian | Ghi chú |
|---|---|---|---|
| 1 | **Problem & Motivation** | 1 phút | Bắt đầu bằng câu hỏi năng lượng |
| 2 | **Overall Framework** (gộp Slide 3+4) | 1.5 phút | Teacher → Student, ANN Attention + SNN FFN |
| 3 | **Tier 1 vs Tier 2 (gộp Slide 5+6)** | 1.5 phút | Tập trung vào khác biệt gradient + loss |
| 4 | **Ablation Design** (Slide 9) | 1 phút | Bảng 4 chiều ablation |
| 5 | **Energy Measurement** (Slide 10) | 1 phút | Dual energy: theoretical vs GPU |
| 6 | **Key Contributions** (Slide 13) | 1 phút | 5 bullets đọc nhanh |

**Bỏ:** Slide Research Gap, Tier 3, Training Pipeline, Expected Results, Risks

**Gợi ý backup slide:** Để slide Risks & Limitations làm backup Q&A.

---

## H. Sơ đồ Quan hệ Giữa Các File

```
Shared Codebase (giống nhau hoàn toàn):
├── PhoBERTTeacher          # Giống nhau trong cả 4 file
├── ArctanSpike             # Surrogate gradient (dùng trong cả SpikeBERTFFN và EquilibriumFFN)
├── SpikeBERTFFN            # Fixed T-step LIF
├── SpikeBERTStudent        # Dùng trong Tier2 A/B/C
├── EquilibriumFFN          # Convergence-based LIF
├── EquilibriumPhoBERTStudent # Dùng trong Tier1 (và Tier3 nếu cấu hình đúng)
├── feature_alignment_loss  # MSE trên hidden states
├── embedding_alignment_loss# MSE trên embedding layer
├── kd_logits_loss          # KL-div soft targets
├── estimate_energy()       # Theoretical energy
└── measure_gpu_energy()    # pynvml actual energy

Điểm khác biệt giữa 4 file:
├── tier1: EXPERIMENT_KIND="tier1" → EquilibriumStudent, always Stage1, full loss
├── tier2_a: EXPERIMENT_KIND="tier2", VARIANT="a" → SpikeBERTStudent, no Stage1, logit-only
├── tier2_b: EXPERIMENT_KIND="tier2", VARIANT="b" → SpikeBERTStudent, no Stage1, +CLS alignment
└── tier2_c: EXPERIMENT_KIND="tier2", VARIANT="c" → SpikeBERTStudent, Stage1 Wiki, full loss
```

---

*Tài liệu được tạo tự động từ phân tích code. Cập nhật khi có kết quả thực nghiệm.*
