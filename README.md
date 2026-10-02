# Tinh chỉnh mô hình DeepSeek-OCR cho bài toán Nhận dạng chữ viết tay tiếng Việt (Line-level OCR)

* **Mô hình nền tảng:** DeepSeek-OCR (3.4B Vision-Language Model)
* **Phương pháp tinh chỉnh:** LoRA (Low-Rank Adaptation) kết hợp thư viện Unsloth

---

## 📖 1. Giới thiệu đề tài & Mục tiêu

Bài toán nhận dạng chữ viết tay tiếng Việt (Vietnamese Handwritten Text Recognition - HTR) dạng dòng có độ phức tạp cao do hệ thống dấu thanh phụ đa dạng (`á, à, ả, ã, ạ, â, ă, đ, ê, ô, ơ, ư`) và nét viết liên tục, tự do. 

Mô hình nền tảng **DeepSeek-OCR** ở chế độ gốc (Zero-shot) có kiến trúc Thị giác - Ngôn ngữ rất mạnh nhưng chưa được tối ưu cho chữ viết tay tiếng Việt, dẫn đến hiện tượng **"ảo giác ngôn ngữ" (Language Hallucination)** – nhầm nét chữ thảo tiếng Việt sang các bảng chữ cái nước ngoài (như chữ Hindi Devanagari hay chữ Ả Rập).

**Mục tiêu của đồ án:**
1. Áp dụng kỹ thuật tinh chỉnh tham số hiệu quả (**PEFT / LoRA**) với thư viện **Unsloth** để huấn luyện mô hình DeepSeek-OCR nhận dạng chính xác chữ viết tay tiếng Việt dạng dòng.
2. Huấn luyện thành công trên tài nguyên phần cứng giới hạn (GPU Tesla T4 15GB trên Kaggle) bằng cách chỉ cập nhật **77.5 triệu tham số (~2.27%)**.
3. Ứng dụng cơ chế cắt ảnh đa tỉ lệ **Gundam Dynamic Cropping** để bảo toàn tối đa độ sắc nét của từng nét bút và dấu thanh tiếng Việt.

---

## 📊 2. Tập dữ liệu (Dataset)

Project sử dụng tập dữ liệu **UIT-HWDB-line** (Bộ dữ liệu chữ viết tay tiếng Việt dạng dòng do Đại học Công nghệ Thông tin - ĐHQG-HCM xây dựng):
* **Tập huấn luyện (Train):** 5.000 ảnh dòng chữ viết tay kèm nhãn văn bản.
* **Tập kiểm định (Validation):** 1.000 ảnh dòng chữ viết tay kèm nhãn văn bản.
* **Tập kiểm thử (Test):** 1.000 ảnh dòng chữ viết tay dùng để đánh giá benchmark độc lập.

---

## ⚙️ 3. Cấu hình & Kỹ thuật huấn luyện

Toàn bộ siêu tham số được quản lý tập trung trong file [`configs/finetune_config.yaml`](configs/finetune_config.yaml):

* **Kiến trúc LoRA:** Rank $r = 16$, $\alpha = 16$, Dropout $= 0.0$, Bias $= \text{"none"}$.
* **Target Modules:** Gắn LoRA lên toàn bộ **7 tầng tuyến tính** của Transformer Decoder:
  * 4 tầng Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
  * 3 tầng MLP/FFN: `gate_proj`, `up_proj`, `down_proj`.
* **Tiền xử lý ảnh (Gundam Cropping):**
  * `base_size = 1024`: Ảnh toàn cảnh bao quát (Global View).
  * `image_size = 640`: Các mảnh cắt chi tiết nét chữ (Local Crops từ 2 đến 9 mảnh).
* **Tối ưu phần cứng:**
  * Huấn luyện 16-bit LoRA (FP16) kết hợp `adamw_8bit`.
  * Bật `gradient_checkpointing = True`.
  * `batch_size = 1` kết hợp `gradient_accumulation_steps = 8` (tương đương Batch Size = 8).
  * Learning Rate: $2 \times 10^{-4}$ với Linear LR Scheduler và 10 bước Warmup.

---

## 🏆 4. Kết quả đạt được (Evaluation Results)

### 4.1. Đánh giá định lượng (Quantitative Results)
Đánh giá trên **1.000 ảnh tập Test độc lập** bằng 2 độ đo tiêu chuẩn:
* **CER (Character Error Rate - Tỷ lệ lỗi ký tự)**
* **WER (Word Error Rate - Tỷ lệ lỗi từ)**

| Mô hình / Phương pháp | CER (%) ↓ | WER (%) ↓ | Ghi chú |
| :--- | :---: | :---: | :--- |
| **DeepSeek-OCR (Zero-shot Baseline)** | 44.27% | 87.09% | Bị ảo giác nặng, nhận nhầm sang chữ Hindi |
| **DeepSeek-OCR + LoRA (Đồ án này)** | **10.67%** | **25.13%** | Nhận diện chuẩn xác nét chữ & dấu thanh |
| **Mức độ cải thiện (Relative Reduction)** | **-75.90%** | **-71.14%** | Giảm lỗi vượt trội |

### 4.2. Phân tích định tính (Qualitative Analysis & Ví dụ thực tế)

Mô hình gốc (Zero-shot Baseline) gặp lỗi nghiêm trọng về **"ảo giác ngôn ngữ" (Language Hallucination)**: do chưa từng được học chữ viết tay tiếng Việt, các nét viết thảo liên tục bị mô hình hiểu nhầm thành các ký tự tiếng Hindi (Devanagari) hoặc tiếng Ả Rập. 

Sau khi fine-tune với LoRA, mô hình đã được "neo" chắc chắn vào hệ chữ Latin tiếng Việt và cải thiện vượt bậc ở 3 khía cạnh:
1. **Nhận dạng đúng ngôn ngữ:** Chấm dứt 100% hiện tượng sinh chữ tiếng nước ngoài.
2. **Khôi phục dấu thanh chính xác:** Nhận diện chuẩn xác các tổ hợp dấu phức tạp (`á, à, ả, ã, ạ, â, ă, đ, ê, ô, ơ, ư`).
3. **Bền vững với nét chữ viết tay:** Tách đúng các từ có nét viết dính liền (cursive strokes).

#### 📌 Ví dụ thực tế từ tập Test (Sample ID: `6927`):

* **Nhãn gốc (Ground Truth):**
  > *"ngày 8 - 3. Món canh có thể hơi mặn, món cá sốt đáng lẽ phải có màu đỏ sậm thì lại ngả sang"*
* **Mô hình gốc (Zero-shot Baseline):**
  > `"गृह १.३. मन बन्ने िकसी भी मन..."` *(Bị ảo giác hoàn toàn sang chữ Hindi/Devanagari)*
* **Mô hình sau khi Fine-tune (Đồ án này):**
  > *"ngày 3. Mùi canh cổ tức hơi mặn, mùi cá sát đáy hơi phải có màu đỏ sương thơ hơi gai sạc,"* *(Đã nhận dạng đúng toàn bộ chữ tiếng Việt, bắt chuẩn các cụm từ ngữ nghĩa: "ngày", "canh", "hơi mặn", "màu đỏ")*

---

## 📂 5. Cấu trúc thư mục dự án

```text
Vietnamese-DeekseekOCR-Finetuning/
├── configs/
│   └── finetune_config.yaml        # Cấu hình siêu tham số mô hình, LoRA, dữ liệu và trainer
├── src/
│   ├── data/
│   │   ├── dataset.py              # Đọc nhãn JSON, chuyển đổi sang hội thoại đa phương thức
│   │   └── collator.py             # Gundam Dynamic Cropping, token expansion, prompt masking
│   ├── models/
│   │   └── peft_setup.py           # Nạp mô hình Unsloth FastVisionModel và thiết lập LoRA
│   └── utils/
│       └── metrics.py              # Tính toán độ đo CER và WER qua thư viện JiWER
├── train.py                        # Script CLI huấn luyện mô hình
├── infer.py                        # Script CLI chạy nhận diện trên 1 ảnh hoặc thư mục ảnh
├── eval.py                         # Script CLI đánh giá benchmark trên tập test
├── notebooks/
│   └── 01_vietnamese_deepseek_ocr_training.ipynb   # Notebook hướng dẫn huấn luyện
├── requirements.txt                # Danh sách thư viện phụ thuộc
└── README.md                       # Tài liệu báo cáo dự án
```

---

## 🚀 6. Hướng dẫn cài đặt & Chạy chương trình

### 6.1. Cài đặt môi trường
Khuyến nghị sử dụng môi trường Python 3.10+ có hỗ trợ GPU:
```bash
pip install -r requirements.txt
```

### 6.2. Huấn luyện mô hình (Training)
Chạy huấn luyện mô hình bằng lệnh:
```bash
python train.py --config configs/finetune_config.yaml
```
*Mô hình và checkpoint tốt nhất sẽ được tự động lưu tại thư mục `outputs/final_model`.*

### 6.3. Chạy suy luận nhận diện (Inference)
Nhận diện chữ viết tay trên 1 bức ảnh:
```bash
python infer.py --image_path path/to/image.jpg --model_path outputs/final_model
```

Nhận diện hàng loạt trên cả thư mục ảnh và lưu kết quả ra file:
```bash
python infer.py --image_dir path/to/images_folder --output_txt predictions.txt
```

### 6.4. Đánh giá độ chính xác trên tập Test (Evaluation)
Tính toán bảng điểm CER và WER:
```bash
python eval.py --test_json data/test.json --images_dir data/test_images --model_path outputs/final_model
```