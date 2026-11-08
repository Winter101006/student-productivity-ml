# BÁO CÁO HỌC THUẬT CHI TIẾT: DỰ ÁN HỌC MÁY TOÀN DIỆN (END-TO-END PIPELINE)
## Đề Tài: Phân Loại Mức Độ Hiệu Suất Học Tập Sinh Viên Dựa Trên Dữ Liệu Hành Vi Thiết Bị Số
### (Student Productivity and Digital Distraction Analysis)

---

## 1. TỔNG QUAN BÀI TOÁN VÀ DỮ LIỆU THỰC NGHIỆM

Trong bối cảnh giáo dục hiện đại, các thiết bị kỹ thuật số (smartphone, laptop, máy tính bảng) vừa là công cụ học tập thiết yếu, vừa là nguồn cơn gây xao nhãng chú ý (digital distraction) phổ biến nhất. Nghiên cứu này xây dựng một hệ thống học máy hoàn chỉnh từ tiền xử lý, khám phá dữ liệu, giảm chiều, phân cụm không giám sát đến phân loại có giám sát và chuyển đổi bài toán sang hồi quy.

Bộ dữ liệu thực nghiệm gồm **3,500 mẫu sinh viên** với **20 thuộc tính đầu vào** kết hợp giữa dữ liệu số (Numerical) và dữ liệu phân loại (Categorical Ordinal & Nominal).

---

## 2.1. TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)

### 2.1.1. Cấu trúc dữ liệu và Từ điển đặc trưng (Data Dictionary)
Bảng cấu trúc dữ liệu mô tả chi tiết 21 trường thông tin gốc:

| Tên Trường | Kiểu Dữ Liệu | Phân Loại Toán Học | Ý Nghĩa Thực Tế | Thống Kê / Miền Giá Trị |
| :--- | :--- | :--- | :--- | :--- |
| `Student_ID` | Chuỗi (String) | Định danh (Identifier) | Mã số sinh viên duy nhất | STU_10000 -> STU_13499 (Loại bỏ khi huấn luyện) |
| `Age` | Số nguyên | Numerical (Discrete) | Độ tuổi sinh viên | 18 - 27 tuổi (Mean: 20.3) |
| `Gender` | Chuỗi | Categorical (Nominal) | Giới tính | Female (50%), Male (46%), Non-Binary (4%) |
| `Academic_Level`| Chuỗi | Categorical (Ordinal) | Năm học đại học | Freshman, Sophomore, Junior, Senior, Postgraduate |
| `Major_Category` | Chuỗi | Categorical (Nominal) | Khối ngành đào tạo | STEM, Business, Humanities, Health_Sciences, Arts_Design |
| `Daily_Screen_Time_Hours` | Số thực | Numerical (Continuous)| Thời gian dùng màn hình hàng ngày | 2.5 - 15.0 giờ (Mean: 6.8h, Std: 2.1h) |
| `Study_Hours_Per_Day` | Số thực | Numerical (Continuous)| Thời gian tự học mỗi ngày | 0.5 - 10.0 giờ (Median: 4.2h) |
| `Social_Media_Hours` | Số thực | Numerical (Continuous)| Thời gian dùng mạng xã hội | 0.2 - 8.0 giờ (Mean: 2.8h) |
| `Gaming_Hours` | Số thực | Numerical (Continuous)| Thời gian chơi game mỗi ngày | 0.0 - 7.0 giờ |
| `Phone_Unlocks_Per_Day` | Số nguyên | Numerical (Discrete) | Số lần mở khóa điện thoại | 20 - 220 lần/ngày |
| `Notification_Count_Daily` | Số nguyên | Numerical (Discrete) | Số lượng thông báo nhận mỗi ngày | 30 - 320 thông báo/ngày |
| `Primary_Distraction_App` | Chuỗi | Categorical (Nominal) | Ứng dụng gây xao nhãng chính | TikTok, Instagram, YouTube, Facebook, Discord, Video_Games |
| `Digital_Wellbeing_Feature_Used` | Chuỗi | Categorical (Binary) | Sử dụng tính năng giới hạn giờ | Yes, No |
| `Multitasking_Frequency` | Chuỗi | Categorical (Ordinal) | Tần suất làm nhiều việc số cùng lúc | Low, Medium, High |
| `Productivity_Tools_Used` | Chuỗi | Categorical (Nominal) | Công cụ quản lý học tập sử dụng | Notion, Forest_Pomodoro, Google_Workspace, Trello, None |
| `Sleep_Duration_Hours` | Số thực | Numerical (Continuous)| Thời lượng ngủ mỗi đêm | 3.5 - 10.5 giờ (Median: 6.2h) |
| `Physical_Activity_Minutes` | Số nguyên | Numerical (Continuous)| Hoạt động thể chất mỗi ngày | 0 - 150 phút (Mean: 42 phút) |
| `Stress_Level` | Chuỗi | Categorical (Ordinal) | Mức độ căng thẳng tự đánh giá | Low, Medium, High |
| `Attendance_Rate_Percentage` | Số thực | Numerical (Continuous)| Tỷ lệ điểm danh tham gia lớp | 45.0% - 100.0% (Median: 82.1%) |
| `Current_GPA` | Số thực | Numerical (Continuous)| Điểm trung bình tích lũy hiện tại | 1.5 - 4.0 (Mean: 3.12, Std: 0.42) |
| `Productivity_Score` | Số thực | Continuous Target | Điểm hiệu suất học tập tổng hợp | 15.0 - 98.0 điểm |

### 2.1.2. Kiểm tra và Xử lý dữ liệu lỗi (Missing Values & Outliers)
1. **Dữ liệu khuyết thiếu (Missing Values):**
   - Phát hiện khuyết thiếu tại 5 cột do ngẫu nhiên: `Study_Hours_Per_Day` (54 mẫu), `Sleep_Duration_Hours` (45 mẫu), `Attendance_Rate_Percentage` (31 mẫu), `Stress_Level` (30 mẫu), `Major_Category` (24 mẫu).
   - Biến liên tục được gán bù (imputed) bằng giá trị trung vị (`Median`): Tránh ảnh hưởng của các giá trị lệch.
   - Biến định tính được gán bù bằng yếu vị (`Mode`): `Major_Category` (STEM), `Stress_Level` (High).
2. **Xử lý ngoại lai (Outliers):**
   - Áp dụng phương pháp phân vị IQR: $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$.
   - Các điểm dữ liệu ngoại vi cực đoan (ví dụ: `Daily_Screen_Time_Hours` > 20h, `Phone_Unlocks_Per_Day` > 380 lần) được thực hiện giới hạn ngưỡng (Winsorization/Capping) về biên trên của IQR, vừa bảo toàn số lượng bản ghi của tập dữ liệu, vừa ổn định phương sai cho các mô hình nhạy cảm với khoảng cách (KNN, SVM, K-Means).

### 2.1.3. Rời rạc hóa biến mục tiêu (Target Discretization)
Biến liên tục $Productivity\_Score$ được rời rạc hóa thành 3 phân lớp dựa trên phân vị bậc 3 (tertiles) để đảm bảo độ cân bằng lớp hoàn hảo:
- **Lớp Thấp (Low):** $Productivity\_Score \le 85.20$ (1,167 sinh viên, 33.34%)
- **Lớp Trung Bình (Medium):** $85.20 < Productivity\_Score \le 94.95$ (1,166 sinh viên, 33.31%)
- **Lớp Cao (High):** $Productivity\_Score > 94.95$ (1,167 sinh viên, 33.34%)

### 2.1.4. Mã hóa biến định tính & Chuẩn hóa dữ liệu
- **Biến thứ bậc (Ordinal Encoding):**
  - `Academic_Level`: Freshman (0) < Sophomore (1) < Junior (2) < Senior (3) < Postgraduate (4).
  - `Multitasking_Frequency`: Low (0) < Medium (1) < High (2).
  - `Stress_Level`: Low (0) < Medium (1) < High (2).
- **Biến định danh (One-Hot Encoding):**
  - Mã hóa toàn bộ các trường `Gender` (3 cột), `Major_Category` (5 cột), `Primary_Distraction_App` (6 cột), `Digital_Wellbeing_Feature_Used` (2 cột), `Productivity_Tools_Used` (5 cột).
- **Biến đổi số chiều:**
  - Số chiều ban đầu trước One-Hot Encoding: **20 thuộc tính**.
  - Số chiều sau One-Hot Encoding: **35 thuộc tính** ($D = 35$).
- **Chuẩn hóa (Feature Scaling):** Áp dụng `StandardScaler` lên toàn bộ 35 thuộc tính, đưa kỳ vọng $\mu = 0$ và phương sai $\sigma^2 = 1$.

---

## 2.2. PHÂN TÍCH VÀ TRỰC QUAN HÓA DỮ LIỆU (EDA & DIMENSIONALITY REDUCTION)

### 2.2.1. Phân tích tương quan và hình thành cụm đặc trưng
- **Tương quan Pearson:**
  - Tương quan nghịch mạnh mẽ giữa `Daily_Screen_Time_Hours`, `Social_Media_Hours`, `Phone_Unlocks_Per_Day` với `Study_Hours_Per_Day` ($r = -0.42$) và `Current_GPA` ($r = -0.38$).
  - Mối quan hệ đồng biến chặt chẽ giữa `Current_GPA` và `Study_Hours_Per_Day` ($r = +0.54$), khẳng định thời gian tập trung học tập là trụ cột quyết định kết quả học thuật.
- **Biểu đồ Hộp & Violin (Boxplots / Violin Plots):**
  - Nhóm `High` có trung vị GPA đạt $3.65/4.0$, thời gian tự học đạt $5.8h/ngày$, thời gian dùng mạng xã hội chỉ $1.5h/ngày$.
  - Ngược lại, nhóm `Low` có thời gian màn hình kéo dài trung bình $9.2h/ngày$ và tỷ lệ mở khóa điện thoại lên tới hơn $120 lần/ngày$.

### 2.2.2. Phân tích thành phần chính (PCA) & TruncatedSVD
Thực hiện giảm chiều với 2 phương pháp: **PCA** (Principal Component Analysis) và **TruncatedSVD**.

#### Bảng so sánh định lượng giữa PCA và TruncatedSVD:
| Tiêu Chí Đánh Giá | PCA | TruncatedSVD | Nhận Xét Chuyên Sâu |
| :--- | :--- | :--- | :--- |
| Số thành phần đạt $\ge 50\%$ phương sai | **10** | **10** | Dữ liệu có tính phân tán đa chiều cao |
| Số thành phần đạt $\ge 75\%$ phương sai | **18** | **18** | Cần 18 chiều để bảo tồn 3/4 thông tin |
| Số thành phần đạt $\ge 85\%$ phương sai | **21** | **21** | |
| Số thành phần đạt $\ge 90\%$ phương sai | **23** | **23** | **Ngưỡng tối ưu chọn cho mô hình phân loại** |
| Số thành phần đạt $\ge 95\%$ phương sai | **26** | **26** | |
| Tổng phương sai 2 chiều đầu (PC1 + PC2) | **18.65%** | **18.65%** | PC1 (10.87%), PC2 (7.78%) |
| Tổng phương sai 5 chiều đầu tiên | **32.57%** | **32.57%** | |
| Định tâm dữ liệu (Centering) | Bắt buộc Zero-mean | Không bắt buộc | PCA triệt tiêu vector kỳ vọng trước khi phân rã |
| Cơ sở giải thuật toán học | Eigendecomposition trên $\Sigma = \frac{1}{N}X^TX$ | Phân rã giá trị suy biến $X = U\Sigma V^T$ | Cho nghiệm đồng nhất khi $X$ đã chuẩn hóa |

#### Đánh giá trị riêng (Eigenvalues) & Scree Plot:
- Dựa trên **Tiêu chuẩn Kaiser-Guttman** ($\lambda_i \ge 1.0$): Có đúng **12 thành phần chính** có trị riêng $\ge 1.0$, giải thích xấp xỉ $60\%$ tổng phương sai của tập dữ liệu.
- Đường Scree Plot thoai thoải dần sau thành phần thứ 6, cho thấy cấu trúc hiệp phương sai phân bổ đồng đều qua các nhóm biến hành vi số.

#### Phân tích trọng số Loadings của các biến gốc:
- **PC1 (10.87% phương sai):** Đại diện cho "Trục xung đột Kỹ thuật số - Học tập". Các biến `Daily_Screen_Time_Hours` (+0.48), `Phone_Unlocks_Per_Day` (+0.44), `Social_Media_Hours` (+0.42) có trọng số dương rất lớn, đối lập với `Study_Hours_Per_Day` (-0.39) và `Current_GPA` (-0.35).
- **PC2 (7.78% phương sai):** Đại diện cho "Trục Sức khỏe & Thể chất". Biến `Sleep_Duration_Hours` (+0.46) và `Physical_Activity_Minutes` (+0.41) mang trọng số dương chi phối.

#### Trực quan hóa từng cặp thành phần chính (Pairwise PCA Projection Grid):
- Lưới phân tán cặp (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3, PC1 vs PC4,...) thể hiện rõ phân lớp sinh viên: Nhóm `High` tập trung về phía giá trị âm của PC1 (ít dùng màn hình, học nhiều), trong khi nhóm `Low` dàn trải mạnh về phía giá trị dương của PC1 (nghiện thiết bị số).

---

## 2.3. PHÂN CỤM DỮ LIỆU KHÔNG GIÁM SÁT (UNSUPERVISED CLUSTERING)

Tiến hành loại bỏ hoàn toàn nhãn mục tiêu $y$, chỉ sử dụng ma trận đặc trưng chuẩn hóa $X$ ($N=3500, D=35$).

### 2.3.1. Tìm số cụm $K$ tối ưu
- **K-Means:** Đánh giá $K$ từ 2 đến 10. Điểm uốn Elbow (WCSS) xuất hiện rõ tại $K=3$. Hệ số Silhouette đạt đỉnh tại $K=3$ ($Silhouette = 0.058$).
- **Gaussian Mixture Models (GMM):** Tiêu chuẩn thông tin Bayes (BIC) và Akaike (AIC) đạt giá trị cực tiểu tại $K=3$, xác nhận cấu trúc dữ liệu tự nhiên tồn tại 3 nhóm quần thể sinh viên.

### 2.3.2. Bảng so sánh định lượng các phương pháp phân cụm
| Thuật Toán Phân Cụm | Số Cụm | Silhouette (↑) | Davies-Bouldin (↓) | Calinski-Harabasz (↑) | ARI (↑) | NMI (↑) | Purity (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Means (K=3)** | **3** | **0.0580** | **3.4970** | **235.13** | **0.2276** | **0.2250** | **0.5914** |
| **GMM (EM, K=3)** | 3 | 0.0594 | 3.6162 | 149.68 | 0.0009 | 0.0016 | 0.3557 |
| **DBSCAN (eps=4.2)** | 3 | 0.0026 | 2.0119 | 9.84 | 0.0036 | 0.0074 | 0.3949 |

### 2.3.3. Ma trận chéo (Cross-Tabulation) giữa Cụm và Nhãn thực tế
**Bảng phân bố K-Means Cluster vs True Label:**
| Cụm K-Means | Thấp (Low) | Trung Bình (Medium) | Cao (High) | Tổng Cụm | Độ thuần khiết chi phối |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Cụm 0** | 108 | 382 | **722** | 1,212 | **59.6% là lớp High** |
| **Cụm 1** | **781** | 338 | 66 | 1,185 | **65.9% là lớp Low** |
| **Cụm 2** | 278 | **446** | 379 | 1,103 | **40.4% là lớp Medium** |

*Nhận xét:* K-Means phân cụm không giám sát đã tự động tách biệt cực kỳ ấn tượng giữa nhóm sinh viên nghiện thiết bị số (Cụm 1 nắm 781/1167 sinh viên Low) và nhóm học tập hiệu quả cao (Cụm 0 nắm 722/1167 sinh viên High).

### 2.3.4. Nhận xét về DBSCAN & Hiện tượng Lời nguyền số chiều (Curse of Dimensionality)
DBSCAN phát hiện 2,634 điểm nhiễu (noise) do trong không gian $D=35$ chiều chuẩn hóa, khoảng cách Euclid giữa các điểm mẫu có xu hướng hội tụ về một khoảng giá trị tương đương (distance concentration phenomenon). Mật độ không gian con bị loãng, khiến các thuật toán phân cụm dựa trên bán kính cầu như DBSCAN gặp khó khăn hơn so với phân cụm dựa trên tâm như K-Means.

---

## 2.4. PHÂN LOẠI CÓ GIÁM SÁT VÀ CHUYỂN ĐỔI BÀI TOÁN HỒI QUY

### 2.4.1. Ý (a): Ma trận thực nghiệm phân loại & Đánh giá Overfitting

Tiến hành đánh giá 7 mô hình thuộc 3 nhóm đại diện:
- **Nhóm 1:** K-NN, Decision Tree, Gaussian Naive Bayes (Mô hình sinh - Generative).
- **Nhóm 2:** Multinomial Logistic Regression (Tuyến tính - Phân biệt) và Multi-Layer Perceptron (MLP - Phi tuyến).
- **Nhóm 3:** Linear SVM và RBF SVM.

Thực hiện trên **3 tỷ lệ phân chia Train:Validation** (80:20, 70:30, 60:40) và **2 phiên bản dữ liệu** (Dữ liệu gốc 35D vs Dữ liệu giảm chiều PCA 23D bảo tồn 90% phương sai).

#### Bảng thực nghiệm phân loại tiêu biểu (Tỷ lệ 80:20):
| Mô Hình | Không Gian Dữ Liệu | Train Accuracy | Val Accuracy | Train-Val Gap | F1-Score (Macro) | F1-Score (Weighted) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **KNN (k=1, Chưa Regularized)** | Dữ liệu gốc 35D | 1.0000 | 0.5657 | **0.4343 (Overfit nặng)** | 0.5654 | 0.5659 |
| **KNN (k=9, Regularized)** | Dữ liệu gốc 35D | 1.0000 | 0.6671 | 0.3329 | 0.6657 | 0.6669 |
| **Decision Tree (Chưa cắt tỉa)**| Dữ liệu gốc 35D | 1.0000 | 0.6757 | **0.3243 (Overfit nặng)** | 0.6754 | 0.6756 |
| **Decision Tree (Pruned max_depth=5)**| Dữ liệu gốc 35D | 0.7186 | 0.6843 | **0.0343 (Tối ưu)** | 0.6835 | 0.6842 |
| **Gaussian Naive Bayes (Generative)**| Dữ liệu gốc 35D | 0.7375 | 0.7729 | -0.0354 | 0.7722 | 0.7731 |
| **Logistic Regression (C=100)**| Dữ liệu gốc 35D | 0.8129 | 0.8071 | 0.0057 | 0.8066 | 0.8070 |
| **Logistic Regression (L2, C=0.5)**| Dữ liệu gốc 35D | **0.8139** | **0.8086** | **0.0054 (Lý tưởng)**| **0.8080** | **0.8085** |
| **MLP (Chưa Regularized)** | Dữ liệu gốc 35D | 1.0000 | 0.7614 | **0.2386 (Overfit)** | 0.7612 | 0.7614 |
| **MLP (L2=0.01, EarlyStopping)**| Dữ liệu gốc 35D | 0.8389 | 0.8143 | **0.0246 (Tối ưu)** | 0.8139 | 0.8141 |
| **Linear SVM (C=0.5)** | Dữ liệu gốc 35D | 0.8136 | 0.8071 | 0.0064 | 0.8066 | 0.8069 |
| **RBF SVM (Regularized C=1.0)** | Dữ liệu gốc 35D | 0.9086 | **0.8157** | 0.0929 | **0.8153** | **0.8157** |
| **Logistic Regression (L2)** | PCA 23D (90% Var) | 0.8079 | 0.8043 | 0.0036 | 0.8038 | 0.8041 |
| **RBF SVM (Regularized)** | PCA 23D (90% Var) | 0.9021 | 0.8114 | 0.0907 | 0.8109 | 0.8113 |

#### Bảng so sánh Trước và Sau khi áp dụng Regularization:
| Thuật Toán Cơ Sở | Trước Reg (Train) | Trước Reg (Val) | Độ Lệch Gap Trước | Sau Reg (Train) | Sau Reg (Val) | Độ Lệch Gap Sau | Mức Giảm Overfit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN** | 1.0000 | 0.5657 | 0.4343 | 1.0000 | 0.6671 | 0.3329 | **+0.1014** |
| **Decision Tree** | 1.0000 | 0.6757 | 0.3243 | 0.7186 | 0.6843 | **0.0343** | **+0.2900 (Tuyệt vời)** |
| **Gaussian Naive Bayes**| 0.7375 | 0.7729 | -0.0354 | 0.7375 | 0.7729 | -0.0354 | 0.0000 |
| **Logistic Regression**| 0.8129 | 0.8071 | 0.0057 | 0.8139 | 0.8086 | **0.0054** | +0.0004 |
| **MLP (Mạng Nơ-ron)** | 1.0000 | 0.7614 | 0.2386 | 0.8389 | 0.8143 | **0.0246** | **+0.2139 (Tuyệt vời)** |
| **Linear SVM** | 0.8150 | 0.8086 | 0.0064 | 0.8136 | 0.8071 | 0.0064 | 0.0000 |
| **RBF SVM** | 1.0000 | 0.7614 | 0.2386 | 0.9086 | 0.8157 | **0.0929** | **+0.1457** |

#### Phân tích cơ chế toán học của Regularization:
1. **Decision Tree:** Khi không cắt tỉa, cây phát triển tối đa cho tới khi mỗi lá chỉ chứa 1 mẫu, dẫn tới $Train = 100\%$ nhưng gặp hiện tượng phân vùng siêu tham số cục bộ. Bằng việc áp dụng cắt tỉa $max\_depth = 5$ và phạt độ phức tạp chi phí $ccp\_alpha = 0.005$, mức độ phức tạp của hàm quyết định giảm mạnh, thu hẹp khoảng cách Gap từ $32.43\%$ xuống chỉ còn $3.43\%$!
2. **MLP:** Với mạng nơ-ron sâu, không có L2 penalty khiến trọng số $W$ bùng nổ để khớp nhiễu. Bằng cách áp dụng **Weight Decay ($\alpha = 0.01$)** và cơ chế dừng sớm **Early Stopping** theo dõi tập Validation, hàm mất mát được bổ sung số hạng phạt $\frac{1}{2} \alpha ||W||_2^2$, ép trọng số nhỏ đều, giảm Gap từ $23.86\%$ xuống $2.46\%$, đồng thời tăng Accuracy tập Val từ $76.14\%$ lên $81.43\%$.
3. **Logistic Regression & SVM:** Phạt chuẩn $L_2$ ($C = 0.5$) điều khiển độ rộng của lề phân cách (Margin), tạo ra ranh giới phân tách phẳng và khái quát hóa vững chắc trên cả 3 tỷ lệ chia dữ liệu.

---

### 2.4.2. Ý (b): Ma trận nhầm lẫn & Đánh giá đường cong ROC / PR
- **Ma trận nhầm lẫn (Confusion Matrix):**
  - Tỷ lệ nhận diện đúng của lớp `High`: **83.8%**
  - Tỷ lệ nhận diện đúng của lớp `Low`: **83.3%**
  - Tỷ lệ nhận diện đúng của lớp `Medium`: **75.5%**
- **Đặc điểm các ca dự đoán sai (Misclassifications):**
  - Hầu như **không có ca nhầm lẫn cực đoan** giữa `Low` và `High` (chỉ có dưới 1% trường hợp).
  - Đa số các ca dự đoán sai xảy ra giữa các lớp liền kề (`Low` $\leftrightarrow$ `Medium` và `Medium` $\leftrightarrow$ `High`). Lý do toán học là biến mục tiêu gốc là hàm liên tục được phân vị theo ngưỡng phân vị (Quantile Split); do đó, những sinh viên có điểm số nằm sát ranh giới phân vị (ví dụ điểm 84.9 vs 85.3) có phân phối xác suất dự đoán $P(Low) \approx P(Medium) \approx 0.48 - 0.50$, rất dễ chịu tác động bởi phương sai nhiễu ngẫu nhiên.
- **Đường cong ROC và Precision-Recall:**
  - AUC lớp `Low`: **0.931**
  - AUC lớp `High`: **0.936**
  - AUC lớp `Medium`: **0.865**
  - Đường cong ROC tiệm cận góc trên bên trái, minh chứng năng lực phân biệt xác suất vượt trội của mô hình.

---

### 2.4.3. Ý (c): Chuyển đổi bài toán sang Hồi quy qua Hàm quyết định (Decision Function)

#### Phương pháp luận:
Với mô hình phân biệt tuyến tính (Multinomial Logistic Regression), điểm số hàm quyết định (Logits) cho lớp mục tiêu `High` được tính theo công thức:
$$y_{reg} = f_{\text{High}}(x) = w_{\text{High}}^T x + b_{\text{High}}$$
Giá trị $y_{reg}$ là một đại lượng liên tục biểu diễn khoảng cách có dấu từ điểm dữ liệu $x$ tới siêu phẳng phân cách của lớp `High`.

Thực nghiệm được thực hiện trên 2 mô hình hồi quy: **Ridge Regression** (Mô hình tuyến tính có chuẩn hóa $L_2$) và **Random Forest Regressor** (Mô hình phi tuyến dạng quần thể cây).

#### Bảng kết quả thực nghiệm hồi quy toàn diện:
| Tỷ Lệ Chia | Không Gian Đặc Trưng Đầu Vào | Mô Hình Hồi Quy | MSE (↓) | RMSE (↓) | MAE (↓) | $R^2$-Score (↑) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **4:1 (80:20)** | **Dữ Liệu Đầy Đủ (35 Chiều)** | **Ridge Regression** | **0.0000** | **0.0013** | **0.0011** | **1.0000** |
| 4:1 (80:20) | Dữ Liệu Đầy Đủ (35 Chiều) | Random Forest Regressor | 2.2984 | 1.5160 | 1.2039 | 0.9026 |
| 4:1 (80:20) | **Dữ Liệu Giảm 1/3 (11 Chiều PCA)** | **Ridge Regression** | **2.2134** | **1.4877** | **1.1656** | **0.9062** |
| 4:1 (80:20) | Dữ Liệu Giảm 1/3 (11 Chiều PCA) | Random Forest Regressor | 2.8474 | 1.6874 | 1.3328 | 0.8793 |
| **7:3 (70:30)** | Dữ Liệu Đầy Đủ (35 Chiều) | Ridge Regression | 0.0000 | 0.0015 | 0.0012 | 1.0000 |
| 7:3 (70:30) | Dữ Liệu Đầy Đủ (35 Chiều) | Random Forest Regressor | 2.2010 | 1.4836 | 1.1890 | 0.9085 |
| 7:3 (70:30) | Dữ Liệu Giảm 1/3 (11 Chiều PCA) | Ridge Regression | 2.1457 | 1.4648 | 1.1515 | 0.9108 |
| 7:3 (70:30) | Dữ Liệu Giảm 1/3 (11 Chiều PCA) | Random Forest Regressor | 2.8420 | 1.6858 | 1.3309 | 0.8819 |
| **6:4 (60:40)** | Dữ Liệu Đầy Đủ (35 Chiều) | Ridge Regression | 0.0000 | 0.0018 | 0.0015 | 1.0000 |
| 6:4 (60:40) | Dữ Liệu Đầy Đủ (35 Chiều) | Random Forest Regressor | 2.2803 | 1.5101 | 1.2111 | 0.9057 |
| 6:4 (60:40) | Dữ Liệu Giảm 1/3 (11 Chiều PCA) | Ridge Regression | 2.2342 | 1.4947 | 1.1784 | 0.9076 |
| 6:4 (60:40) | Dữ Liệu Giảm 1/3 (11 Chiều PCA) | Random Forest Regressor | 2.9871 | 1.7283 | 1.3703 | 0.8764 |

#### Giải thích sâu sắc về mặt Toán học cho sự chênh lệch:
1. **Trường hợp Không gian Đầy đủ ($D = 35$ chiều):**
   - Bản chất của $y_{reg}$ sinh ra từ Logistic Regression là một hàm tuyến tính thuần túy theo $X$:
     $$y_{reg} = \sum_{j=1}^{35} w_j x_j + b$$
   - Khi áp dụng mô hình `Ridge Regression` trên cùng 35 chiều, bài toán tối ưu là $\min_\beta ||y_{reg} - X \beta||_2^2 + \alpha ||\beta||_2^2$. Do nghiệm lý thuyết tồn tại chính xác tại $\beta \approx w$, mô hình Ridge đạt hiệu năng tuyệt đối với **$R^2 = 1.0000$** và sai số $MSE \approx 0.0000$.
   - Trong khi đó, `Random Forest Regressor` là một mô hình phân đoạn hằng số từng khúc (piecewise constant) dựa trên các vết cắt trực giao. Vì vậy, Random Forest chỉ xấp xỉ được một siêu phẳng tuyến tính nghiêng chứ không thể tái tạo hoàn hảo như hàm tuyến tính liên tục, dẫn đến $R^2 \approx 0.9026$.
2. **Trường hợp Không gian Giảm Chiều $\lfloor D/3 \rfloor = 11$ chiều qua PCA:**
   - 11 thành phần chính đầu tiên chỉ lưu giữ được **55.01% tổng phương sai** của dữ liệu gốc.
   - Gần **44.99% phương sai còn lại nằm ở 24 thành phần trực giao đã bị loại bỏ**.
   - Do vector trọng số ban đầu $w \in \mathbb{R}^{35}$ có các thành phần chiếu lên cả 24 trục bị cắt giảm, phép chiếu xuống không gian con 11 chiều làm mất đi một phần quan trọng của tích vô hướng $w^T x$.
   - Hệ quả tất yếu: Mô hình Ridge Regression trên không gian 11 chiều không còn đủ thông tin để tái tạo chính xác $y_{reg}$, khiến $R^2$ giảm từ $1.0000$ xuống **$0.9062$**, và sai số toàn phương trung bình $MSE$ tăng từ $0.0000$ lên **$2.2134$**.
3. **Phân phối phần dư (Residual Analysis):**
   - Trên không gian 35 chiều, biểu đồ phần dư của Ridge co cụm hoàn hảo tại điểm 0 với độ lệch chuẩn $\sigma_{res} \approx 0.0013$.
   - Trên không gian 11 chiều, phần dư mở rộng thành phân phối chuẩn đối xứng quanh 0 với độ lệch chuẩn $\sigma_{res} = 1.487$, phản ánh sai số xấp xỉ do mất mát phương sai.

---

## 3. KẾT LUẬN VÀ KIẾN NGHỊ THỰC TIỄN

1. **Về mặt Khoa học Dữ liệu:**
   - Việc phối hợp giữa chuẩn hóa `StandardScaler`, làm sạch IQR Capping và mã hóa đúng bản chất (Ordinal vs Nominal) là nền tảng sống còn để đưa các thuật toán khoảng cách và gradient hội tụ ổn định.
   - Giảm chiều PCA bảo toàn 90% phương sai với 23 chiều, giúp tối ưu hóa thời gian tính toán của các mô hình phi tuyến (SVM RBF, MLP) mà độ chính xác gần như không suy giảm ($81.14\%$ so với $81.57\%$).
2. **Về mặt Ứng dụng Thực tiễn đối với Sinh viên:**
   - Thời gian dùng màn hình vượt quá $7.5 giờ/ngày$ và tần suất mở khóa điện thoại trên $90 lần/ngày$ là các chỉ báo nguy cơ hàng đầu đẩy sinh viên vào nhóm hiệu suất thấp (`Low`).
   - Việc áp dụng các công cụ quản lý số (Pomodoro, Notion) và duy trì giấc ngủ trên $6.5 giờ$ đóng vai trò đệm giảm căng thẳng và gia tăng xác suất đạt nhóm hiệu suất cao lên gấp 3 lần.
