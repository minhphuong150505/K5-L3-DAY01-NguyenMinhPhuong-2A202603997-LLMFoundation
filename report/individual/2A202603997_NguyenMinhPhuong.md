# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603997
- **Họ và tên:** Nguyễn Minh Phương
- **SEED** (in ở ô đầu notebook): 7114

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Bộ kiểm tra trong notebook: **15/15 đạt**, đủ 10/10 dự đoán. Notebook chạy trên máy cá nhân có GPU (RTX 5060 Ti), LLM provider là **Groq** (`openai/gpt-oss-120b` / `openai/gpt-oss-20b`).

## 2. Kết quả chính

| Phương án | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.899 | 0.857 | 0.021 | 0 |
| 2. BoW + MLP (deep learning) | 0.940 | 0.947 | 0.044 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.890 | 0.947 | 7.307 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.912 | 0.919 | 2.210 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.915 | 0.919 | 2.403 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-oss-20b) | — | 0.900 | 5550 | 0.005 |
| 4b. LLM lớn zero-shot (gpt-oss-120b) | — | 0.947 | 9158 | 0.011 |
| 4c. LLM nhỏ few-shot | — | 0.947 | 3682 | 0.016 |

Trên đường Pareto (1 − F1 so với p95): BoW + LogReg và BoW + MLP. Đạt SLA p95 < 50 ms: cả 5 mô hình cục bộ, không có LLM nào.

## 3. Dự đoán và thực tế

1. **emb_search**: đoán kết quả gần nhất là *ham* vì truy vấn "complimentary holiday… reply now to claim" không có free/win/prize. Thực tế cả 6 tin gần nhất đều là **spam** (cosine 0.52–0.56). Lệch vì embedding mã hoá *nghĩa* của câu ("được chọn nhận phần thưởng/kỳ nghỉ"), không phụ thuộc vào việc có từ khoá hay không.
2. **tradeoff**: đoán một mô hình embedding sẽ nằm trên đường Pareto. Thực tế chỉ BoW + LogReg và BoW + MLP nằm trên đó; BoW + MLP vừa nhanh hơn (0.04 ms so với 2.2 ms) vừa ít lỗi hơn 3b trên test thường. Lệch vì tập test cùng phân phối với train, nơi từ khoá là đủ; ưu thế của embedding chỉ lộ ra ở 50 case khó.
3. **hard_eval**: đoán LLM còn khoảng 75–85% và khó nhất ở prompt injection. Thực tế LLM nhỏ đạt **88%** (prompt phòng thủ: **92%**); nhóm khó nhất với LLM lại là *Tưởng spam nhưng không phải* (60–70%) và *Nói về spam* (75%), còn prompt injection đạt 83% (4a) và 100% (4d). Lệch vì LLM bị đánh lừa bởi các từ khoá mang hình thức spam ("FREE pizza", "Cash prize") nhiều hơn bởi câu lệnh chèn vào.

## 4. Phân tích 50 case khó

Tụt nhiều nhất là **BoW + LogReg**: F1 từ 0.857 (SUB) xuống **0.378**, accuracy 54%. Nó sai nặng nhất ở nhóm *Thao túng / lừa đảo* (12% đúng), *Né từ khoá* (17%) và *Spam giọng tự nhiên* (20%). Các mô hình embedding giữ được 64–74%, còn LLM giữ được 88–92%.

- **#29** `C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd…`: BoW tách thành các token như `c0ngr4tulat10ns`, `y0u`, `w0n`, không có trong vocab 2000 từ. Cả câu chỉ còn 4 token trong vocab (`a`, `£1000`, `at`, `gift`), vector gần như rỗng, nên mô hình rơi về phía ham theo bias (p(spam) = 0.08). Embedding và LLM vẫn đọc được nhờ tokenizer subword và ngữ nghĩa.
- **#21** `Hi Mum, I dropped my phone in the toilet… send £200 to my friend's account`: không có từ khoá spam nào; các từ "hi", "mum", "my", "i" đều kéo mạnh về ham (p(spam) = 0.03). Spam kiểu lừa đảo mạo danh người thân chỉ nhận ra được qua *ý đồ* (xin chuyển tiền gấp từ số lạ), là thứ BoW không biểu diễn được.

Ngược lại, embedding kNN chỉ đúng 25% ở nhóm *Nói về spam*: tin kể lại "I just got a text saying I won £1000…" có vector nằm sát cụm spam thật.

## 5. Thử thách cá nhân

**Đề #5** — Embedding: viết 4 tin (2 spam né từ khoá, 2 ham chứa "free"/"win"), so BoW với Embedding + LogReg.

| Tin | Nhãn | BoW p(spam) | Enc+LR p(spam) |
| --- | --- | --- | --- |
| Clinic, no-cost health check, gửi ngày sinh | spam | 0.81 ✓ | 0.08 ✗ |
| Monthly customer draw, thẻ mua sắm 500 pound | spam | 0.71 ✓ | 0.86 ✓ |
| Are you free tonight… our team will win | ham | 0.04 ✓ | 0.00 ✓ |
| Didn't win the raffle but got a free coffee | ham | 0.02 ✓ | 0.06 ✓ |

BoW đúng 4/4, Encoder + LogReg đúng 3/4. Embedding sai tin "clinic" vì về nghĩa nó giống một tin dịch vụ/lịch hẹn hợp lệ, không mang ngữ nghĩa trúng thưởng. BoW lại bắt được nhờ các token có trọng số spam ("from", "team", "no", "cost", "check"). Với 2 tin ham, cả hai mô hình đều thấy ngữ cảnh hội thoại lấn át "free"/"win". Chỉ 4 tin nên không kết luận được mô hình nào tốt hơn; bài học là embedding yếu với spam *bắt chước tin hợp lệ*.

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày (~116 tin/s), p95 < 50 ms/tin.

- LLM bị loại khỏi đường xử lý chính: p95 3.7–9 s. Chi phí theo ước tính của notebook là ~$50/ngày (~$1.500/tháng), thực tế cao hơn nhiều (~$200/ngày) khi tính reasoning tokens.
- Chọn **router 2 tầng**. Tầng 1 là **Encoder (MiniLM) + MLP head** (F1 0.915, p95 2.4 ms) hoặc BoW + MLP (F1 0.940, p95 0.04 ms) cho mọi tin. Chỉ tin lưỡng lự (p ∈ [0.3, 0.7]) hoặc bị gắn cờ (leetspeak, mẫu prompt injection) mới chuyển sang tầng 2: LLM nhỏ + prompt phòng thủ (92% đúng trên 50 case khó), xử lý bất đồng bộ.
- Đánh đổi chấp nhận: một phần nhỏ tin bị trễ vài giây, và tốn chi phí LLM cho vài % lưu lượng. Bù lại, phần lớn tin đạt SLA gần như miễn phí, và độ bền với spam kiểu mới cao hơn hẳn chỉ dùng BoW (54–60% trên case khó).

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời/viết gì | Đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 4 | Đọc notebook và sửa lỗi | Bỏ tham số `resoning_effort` (gây SyntaxError, trùng với `provider_kwargs`), trả lại tên secret `GROQ_API_KEY`, khôi phục ô kiểm tra 4.1 + 4.4 gốc | Ô kiểm tra 4.1 + 4.4 và 4.2–4.3 đều ✅ |
| 4 | Thêm giá Groq, chống lỗi 429 | Thêm giá gpt-oss vào `PRICE_PER_1M`, tăng `max_retries` trong `llm_classify_batch` | Chạy thật trên Groq: 0 invalid |
| 4–5 | Chạy toàn bộ notebook trên máy GPU và điền bài | Viết các dự đoán `llm_models`, `llm_fewshot`, `tradeoff`, `hard_eval`, 5 case bonus, code thử thách #5, `HAND_CALC`, các câu trả lời và báo cáo này dựa trên output | 15/15 kiểm tra ✅; đối chiếu từng con số với output notebook |

Điều hiểu ra hôm nay: điểm F1 cao trên tập test cùng phân phối không chứng minh mô hình "hiểu" spam. BoW + MLP đứng đầu bảng test nhưng chỉ đúng 60% trên 50 case khó, trong khi LLM gần như không tụt. Chọn mô hình cho sản phẩm cần nhìn cùng lúc chất lượng trên dữ liệu khó, độ trễ p95 và chi phí, và thường lời giải tốt nhất là kết hợp nhiều cách (router) thay vì một mô hình duy nhất.
