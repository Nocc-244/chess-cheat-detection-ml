# chess-cheat-detection-ml
Machine Learning based Chess Cheat Detection using Stockfish, Random Forest, XGBoost and LightGBM
# HỆ THỐNG PHÁT HIỆN GIAN LẬN TRONG CỜ VUA TRỰC TUYẾN BẰNG HỌC MÁY

## Giới thiệu

Đây là đồ án cuối kỳ môn Học máy với mục tiêu xây dựng hệ thống phát hiện hành vi gian lận trong cờ vua trực tuyến bằng các kỹ thuật Machine Learning.

Hệ thống sử dụng Stockfish để phân tích chất lượng nước đi của người chơi, trích xuất các đặc trưng quan trọng và huấn luyện mô hình nhằm phân loại người chơi thành hai nhóm: bình thường và có dấu hiệu sử dụng công cụ hỗ trợ.
Dữ liệu gốc gồm 2000 ván cờ sạch từ lichess và 2000 ván cờ gian lận từ Kaggle 
Dữ liệu được thu thập từ Kaggle và các nguồn công khai.
Do dung lượng lớn nên không đính kèm toàn bộ dữ liệu gốc.

---

## Bộ dữ liệu

Dữ liệu sử dụng trong nghiên cứu bao gồm:

- 2000 ván cờ sạch (Clean Games)
- 2000 ván cờ có dấu hiệu gian lận (Cheat Games)

Sau quá trình trích xuất đặc trưng cho từng người chơi, tập dữ liệu cuối cùng thu được khoảng 7984 mẫu dữ liệu phục vụ cho việc huấn luyện và đánh giá mô hình.

---

## Các đặc trưng sử dụng

Một số đặc trưng chính được trích xuất từ Stockfish:

- ACPL (Average Centipawn Loss)
- Opening ACPL
- Middlegame ACPL
- Endgame ACPL
- Best Move Rate
- Top-3 Move Rate
- Blunder Rate
- Mistake Rate
- Move Count
- Elo
- Elo Difference

---

## Các mô hình sử dụng

- Random Forest
- XGBoost
- LightGBM

---

## Cấu trúc dự án

- `3_extract_features.py`: Trích xuất đặc trưng bằng Stockfish
- `4_train_model.py`: Huấn luyện và đánh giá mô hình
- `dataset_ml_depth12.csv`: Tập dữ liệu cuối cùng
- `chess_fraud_model.pkl`: Mô hình đã huấn luyện
- `GUI`: Giao diện mô phỏng hệ thống
- `dothichess.png`: Hình ảnh kết quả đánh giá

---

## Kết quả

Kết quả thực nghiệm cho thấy hệ thống đạt hiệu năng tốt trong việc phát hiện gian lận:

- ROC-AUC ≈ 0.90
- Precision ≈ 0.72
- Recall ≈ 0.70
- F1-Score ≈ 0.71

---

## Nhóm thực hiện

Đồ án cuối kỳ học phần Học máy

Trường Đại học Khoa học Tự nhiên - Đại học Quốc gia Hà Nội

Khoa Vật lý

Ngành Kỹ thuật Điện tử và Tin học (EEI)

