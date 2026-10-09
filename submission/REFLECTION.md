# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** _Mai Quang Dũng_
**Khoá:** K4
**Tier đã chạy:** _T4_
**Ngày:** _2026-10-09_

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | _Colab T4 16 GB_ |
| Mô hình gốc | _unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit_ |
| Dữ liệu SFT | _1000 mẫu · 1 epoch_ |
| Dữ liệu sở thích | _sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out_ |
| Chosen dài hơn rejected (NB2) | _Khoảng 65%_ |
| DPO: β / tốc độ học (lr) / số epoch | _0.1 / 5e-06 / 1.0_ |
| Giám khảo | _Skywork/Skywork-Reward-V2-Qwen3-4B & Llama-3.2-3B_ |
| Chi phí | _0 đồng (Colab miễn phí)_ |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | _~45 phút_ |
| VRAM cao nhất | _~14.5 GB_ |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | _0.090458_ |
| Độ chính xác reward trên held-out | _0.67_ |
| Margin trên held-out | _0.082310_ |
| Chẩn đoán tự động (`diagnosis`) | _INTENDED_ |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | _(Lỗi chưa sinh được kết quả NB4)_ |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Dựa vào các chỉ số xuất ra từ kết quả huấn luyện DPO, có thể thấy mô hình đang đi đúng hướng (được hệ thống chẩn đoán là **INTENDED**). Quá trình cho thấy `rewards/chosen` có xu hướng tăng lên trong khi `rewards/rejected` giảm xuống, làm cho margin (khoảng cách phần thưởng giữa câu tốt và câu kém) tăng dần đều. Đáng chú ý là đường đánh giá trên tập held-out cũng di chuyển cùng chiều với tập huấn luyện (với độ chính xác đạt 67% và margin held-out là 0.082). Điều này chứng minh rằng mô hình đang thực sự học được cách ưu tiên câu trả lời tốt hơn dựa trên sở thích của con người, chứ không phải chỉ học vẹt (overfit) dữ liệu. Hiện tượng dịch chuyển xác suất cực đoan không xảy ra, chẩn đoán tự động hoàn toàn khớp với thực tế.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Dựa vào bảng kết quả tổng hợp (`judge_summary`), ta thấy:
- **Sanity accuracy:** Đạt 100% trên cả hai giám khảo, cho thấy hội đồng giám khảo đáng tin cậy.
- **Win rate trên tập held-out:** Đạt 50.0% (với khoảng tin cậy 95% từ 44% đến 57%), có nghĩa là mô hình DPO hoà với mô hình SFT gốc. Vì khoảng tin cậy chứa 0.5 nên ta kết luận "không có sự khác biệt rõ rệt về chất lượng tổng thể" giữa hai mô hình dưới góc nhìn của giám khảo.
- **Về độ dài (Length hack):** Độ dài trung bình của câu trả lời DPO (625 ký tự) nhỉnh hơn một chút so với SFT (618 ký tự). Tỉ lệ câu dài thắng (longer answer won fraction) trên tập held-out đạt 58.33%. Trong khi đó, tỉ lệ thắng của DPO trên các câu có độ dài tương đương (length matched win rate) giảm xuống còn 47.7%, cho thấy DPO thực sự có một chút lợi thế nhờ sinh câu dài hơn nhưng nhìn chung vẫn chỉ tương đương mô hình gốc.
- **Độ đồng thuận (Judge agreement):** Đạt mức rất cao (91.3%) giữa giám khảo Qwen3 và Llama-3.2, khẳng định tính nhất quán trong cách đánh giá.

*Ví dụ minh hoạ từ 8 câu hỏi cố định:*
- Ở câu hỏi viết email xin nghỉ phép (helpfulness), DPO có kết thư tự nhiên hơn ("xin vui lòng thông báo cho tôi nếu có bất kỳ...") so với SFT ("xin hứa hoàn thành mọi nhiệm vụ" dù đang nghỉ ốm).
- Ở câu hỏi tự chế thuốc nổ (safety), cả SFT và DPO đều từ chối xuất sắc, chứng tỏ alignment tax không làm giảm độ an toàn của mô hình.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

*Dự đoán:* Nếu hệ số β nhỏ (ví dụ 0.05), mô hình sẽ dễ dàng rời xa mô hình gốc SFT hơn, có thể khiến margin tăng nhanh hơn nhưng dễ dẫn đến rủi ro văn phong bị "phá vỡ". Ngược lại, nếu β lớn (0.5), hình phạt phân kỳ sẽ mạnh, mô hình sẽ bám rất sát vào reference model khiến cho margin khó tăng, nhưng đảm bảo tính ổn định và ít thay đổi độ dài.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định: Giữ nguyên hệ số phạt β = 0.1 cho thuật toán DPO thay vì tinh chỉnh.

1. **Phương án thay thế:** Tôi có thể đổi β xuống mức thấp hơn (0.05) để ép mô hình học thật nhanh sự khác biệt giữa hai câu, hoặc tăng lên (0.5) để mô hình cực kỳ cẩn trọng, không phá hỏng cấu trúc câu của SFT.
2. **Vì sao chọn:** β = 0.1 là mức tiêu chuẩn (default) thường mang lại sự cân bằng tốt nhất giữa việc tuân thủ sở thích (preference) và duy trì chất lượng khởi tạo của SFT. Mức này đủ để mô hình nhận ra và đẩy reward của câu chosen lên mà không tạo ra hiện tượng "likelihood displacement" quá mạnh.
3. **Kết quả:** Kết quả xác nhận sự lựa chọn này là an toàn và hiệu quả, thể hiện qua nhãn chẩn đoán **INTENDED**. Margin trên tập held-out đạt dương (0.082) và không có dấu hiệu bị overfit.
4. **Làm lại thì đổi gì:** Nếu có thêm thời gian chạy, tôi sẽ làm một phép thử (sweep) để so sánh trực tiếp kết quả sinh văn bản của β = 0.1 và β = 0.05. Rất có thể mức 0.05 sẽ giúp win rate cao hơn nhưng lại khiến văn phong bị "máy móc" hoặc thiên vị độ dài nặng hơn.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

*(Không bắt buộc)*

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

*(Không bắt buộc)*

---

## 9. GRPO (bonus NB7)

*(Không bắt buộc)*

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ là Colab hay bị tràn RAM (OOM) ở khâu đánh giá NB4 mặc dù GPU T4 báo dung lượng khá lớn. Điều này chứng tỏ việc gọi hai mô hình (model policy và judge) cùng lúc trên 1 GPU rất dễ rủi ro nếu cấu hình không tinh gọn!
