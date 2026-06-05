================================================================================
    THÔNG TIN VÀ HƯỚNG DẪN DỰ ÁN DỰ ĐOÁN BỆNH TIM
    Sử dụng tiền xử lý dữ liệu, giảm chiều, phân cụm và phân loại Machine Learning
================================================================================

================================================================================
PHẦN 1: THÔNG TIN THÀNH VIÊN NHÓM
================================================================================

DANH SÁCH THÀNH VIÊN:
1. HOÀNG DANH DŨNG == MSV : 23000103
2. NGUYỄN HỮU DŨNG == MSV : 23000104

PHÂN CÔNG CÔNG VIỆC CHI TIẾT:

1. THÀNH VIÊN THỨ NHẤT
   Trách nhiệm: Mô tả nguồn dữ liệu, phân tích ban đầu, trực quan hóa dữ liệu gốc,
   tiền xử lý dữ liệu, chuẩn hóa dữ liệu, mã hóa dữ liệu, chia train/test, LDA,
   phân cụm DBSCAN, mô hình phân loại KNN và hoàn thiện tài liệu.

   Công việc cụ thể:

   a) Mô tả nguồn dữ liệu và phân tích ban đầu:
      - Đọc bộ dữ liệu heart_2020_cleaned.csv vào notebook.
      - Kiểm tra cấu trúc dữ liệu: số dòng, số cột, tên cột và kiểu dữ liệu.
      - Mô tả bộ dữ liệu Heart Disease 2020.
      - Trình bày ý nghĩa biến mục tiêu HeartDisease.
      - Thống kê số bản ghi, số thuộc tính và phân bố nhãn No/Yes.
      - Nhận xét hiện tượng mất cân bằng lớp trong dữ liệu.
      - Phân tích các nhóm biến: biến số, biến Yes/No, biến thứ tự và biến danh mục.

   b) Trực quan hóa dữ liệu gốc:
      - Vẽ biểu đồ phân bố biến mục tiêu HeartDisease.
      - Vẽ histogram cho các cột số như BMI, PhysicalHealth, MentalHealth, SleepTime.
      - Vẽ biểu đồ boxplot ban đầu để quan sát outlier trên các cột số.
      - Vẽ các biểu đồ theo biến phân loại như Sex, AgeCategory, GenHealth, Race, Diabetic.
      - Trực quan hóa mối quan hệ giữa một số đặc trưng đầu vào và HeartDisease.

   c) Kiểm tra và làm sạch dữ liệu:
      - Kiểm tra giá trị thiếu, dữ liệu trùng lặp và các giá trị không hợp lệ.
      - Xóa các dòng trùng lặp để tạo bộ dữ liệu sạch hơn.
      - Mô tả thống kê dữ liệu sau khi xóa trùng lặp.
      - Xác định các cột số gồm BMI, PhysicalHealth, MentalHealth và SleepTime.
      - Kiểm tra và thống kê số lượng outlier trên các cột số.
      - Xử lý outlier bằng phương pháp capping theo ngưỡng phân vị 1% và 99%.
      - Vẽ boxplot sau khi capping outlier để so sánh với dữ liệu ban đầu.
      - Mô tả lại dữ liệu sau khi capping outlier.

   d) Chia train/test:
      - Tách biến đầu vào X và biến mục tiêu y.
      - Chuyển biến mục tiêu HeartDisease từ No/Yes sang 0/1.
      - Chia dữ liệu theo nhiều tỉ lệ:
        * 80/20
        * 70/30
        * 60/40
      - Sử dụng stratify=y để giữ tỉ lệ lớp No/Yes tương đối ổn định ở train và test.
      - Thực hiện chia train/test trước khi chuẩn hóa và mã hóa để tránh rò rỉ dữ liệu.

   e) Chuẩn hóa riêng các cột số:
      - Chuẩn hóa riêng các cột kiểu số bằng StandardScaler.
      - Đảm bảo scaler chỉ được fit trên tập train để tránh rò rỉ dữ liệu test.
      - Biến đổi tập test bằng scaler đã học từ tập train.
      - Chỉ chuẩn hóa các cột BMI, PhysicalHealth, MentalHealth và SleepTime.
      - Giữ nguyên các cột phân loại ở bước này để mã hóa ở bước sau.

   f) Mã hóa dữ liệu sau khi chuẩn hóa cột số:
      - Mã hóa các cột Yes/No sang 0/1.
      - Mã hóa Sex sang dạng số.
      - Mã hóa AgeCategory theo thứ tự độ tuổi.
      - Mã hóa GenHealth theo thứ tự mức độ sức khỏe.
      - One-hot encoding cho Race và Diabetic.
      - Đồng bộ cột giữa train và test sau one-hot encoding.
      - Mô tả dữ liệu sau chuẩn hóa và mã hóa để kiểm tra dữ liệu đã sẵn sàng cho mô hình.

   g) Giảm chiều bằng LDA:
      - Triển khai Linear Discriminant Analysis với n_components=1.
      - Fit LDA trên tập train đã tiền xử lý và dùng nhãn y_train.
      - Biến đổi train/test sang không gian LD1.
      - Thống kê phân bố LD1 theo từng lớp HeartDisease.
      - Trực quan hóa phân bố LD1 để đánh giá mức độ tách lớp.

   h) Phân cụm DBSCAN:
      - Loại bỏ trường đầu ra HeartDisease khi phân cụm.
      - Chạy DBSCAN với tối đa 50,000 mẫu để giảm thời gian và bộ nhớ.
      - Giảm về không gian 2 chiều bằng PCA để phục vụ DBSCAN và trực quan hóa.
      - Dùng k-distance plot để hỗ trợ lựa chọn eps.
      - Thử nghiệm nhiều giá trị eps và min_samples.
      - Đánh giá phân cụm bằng các độ đo:
        * Silhouette Score
        * Davies-Bouldin Score
        * Calinski-Harabasz Score
        * Adjusted Rand Index
        * Normalized Mutual Information
        * Cluster Purity
      - Lập bảng số lượng mẫu theo cụm và tỉ lệ nhãn HeartDisease trong từng cụm.
      - Trực quan hóa cụm bằng màu sắc và phân biệt nhãn bằng hình dạng điểm.

   i) Mô hình phân loại KNN:
      - Triển khai KNeighborsClassifier.
      - Sử dụng n_neighbors=15, weights='distance', metric Minkowski với p=2.
      - Chạy KNN trên toàn bộ train/test khi cấu hình không giới hạn lấy mẫu.
      - Chạy KNN trên các bộ dữ liệu RAW, PCA và LDA.
      - Đánh giá KNN trên các tỉ lệ split 80/20, 70/30 và 60/40.
      - Ghi nhận Accuracy, Precision, Recall, F1-score, ROC-AUC và overfit gap.

   j) Hoàn thiện tài liệu:
      - Tổng hợp cấu trúc notebook.
      - Viết README hướng dẫn chạy chương trình.
      - Tóm tắt các mô hình đã dùng và vai trò của từng mô hình.
      - Đưa ra nhận xét cuối cùng về pipeline và kết quả thực nghiệm.

2. NGUYỄN HỮU DŨNG
   Trách nhiệm: Các phần còn lại của notebook, gồm PCA,trực quan hóa dữ liệu, Logistic Regression,
   Linear SVM, hồi quy từ score phân lớp và so sánh kết quả.

   Công việc cụ thể:

   a) Giảm chiều bằng PCA:
      - Triển khai PCA trên dữ liệu đã tiền xử lý.
      - Tính explained variance ratio cho từng principal component.
      - Tính cumulative explained variance.
      - Xác định số PC cần thiết để giữ lại ít nhất 90% thông tin.
      - Tạo bảng thống kê cho các thành phần chính đầu tiên.
      - Trực quan hóa explained variance bằng bar chart và cumulative curve.
      - Trực quan hóa các cặp thành phần PC1-PC6.
      - Trực quan hóa mối quan hệ giữa các principal components và HeartDisease.

   b) So sánh PCA và LDA:
      - So sánh PCA là phương pháp không giám sát và LDA là phương pháp có giám sát.
      - So sánh mục tiêu của PCA là bảo toàn phương sai.
      - So sánh mục tiêu của LDA là tăng khả năng tách lớp.
      - Lập bảng tổng hợp các độ đo: explained variance, số component và mức độ tách lớp.

   c) Mô hình Logistic Regression:
      - Triển khai LogisticRegression với class_weight='balanced'.
      - Sử dụng regularization L2 với tham số C=1.0.
      - Tăng max_iter để đảm bảo mô hình hội tụ.
      - Chạy trên dữ liệu RAW, PCA và LDA.
      - Chạy trên các tỉ lệ split 80/20, 70/30 và 60/40.
      - Lấy predict_proba để tính ROC-AUC và phục vụ bài toán hồi quy score.

   d) Mô hình Linear SVM:
      - Triển khai LinearSVC cho bài toán phân loại tuyến tính.
      - Sử dụng class_weight='balanced' để xử lý mất cân bằng lớp.
      - Sử dụng regularization thông qua tham số C.
      - Chạy trên dữ liệu RAW, PCA và LDA.
      - Chạy trên các tỉ lệ split 80/20, 70/30 và 60/40.
      - Lấy decision_function làm score phân lớp.

   e) Đánh giá và trực quan hóa kết quả phân loại:
      - Xây dựng hàm đánh giá chung cho các mô hình phân loại.
      - Tính Accuracy, Precision, Recall, F1-score và ROC-AUC.
      - Tính overfit gap bằng Train Accuracy - Test Accuracy.
      - In classification report.
      - Vẽ confusion matrix cho KNN, Logistic Regression và Linear SVM.
      - Vẽ đường cong ROC-AUC cho KNN, Logistic Regression và Linear SVM trên dữ liệu RAW với split 80/20.
      - So sánh kết quả giữa dữ liệu RAW, PCA và LDA.
      - Nhận xét mô hình có bị overfit hay không.

   f) Hồi quy từ score phân lớp:
      - Chọn score đầu ra từ Logistic Regression và Linear SVM.
      - Chuyển bài toán phân loại thành bài toán hồi quy với biến mục tiêu mới là score.
      - Tạo bộ dữ liệu đầu vào nguyên bản và bộ dữ liệu PCA còn 1/3 số chiều.
      - Triển khai Linear Regression.
      - Triển khai Ridge Regression.
      - Đánh giá bằng MAE, RMSE và R2.
      - Trực quan hóa mối quan hệ giữa score thật và score dự đoán.

CÔNG VIỆC CHUNG CỦA NHÓM:
- Kiểm tra lại notebook trước khi nộp.
- Chạy thử các cell theo đúng thứ tự từ trên xuống dưới.
- Kiểm tra lỗi thư viện, lỗi đường dẫn file và lỗi định dạng dữ liệu.
- So sánh kết quả giữa các mô hình.
- Thống nhất nhận xét và kết luận.

================================================================================
MỤC LỤC
================================================================================
PHẦN 1: THÔNG TIN THÀNH VIÊN NHÓM
PHẦN 2: NGUỒN DỮ LIỆU VÀ CẤU TRÚC DỮ LIỆU
PHẦN 3: TỔ CHỨC VÀ CẤU TRÚC CHƯƠNG TRÌNH
PHẦN 4: CÁC KỊCH BẢN THỰC NGHIỆM
PHẦN 5: KẾT QUẢ VÀ ĐÁNH GIÁ
PHẦN 6: HƯỚNG DẪN SỬ DỤNG

================================================================================
PHẦN 2: NGUỒN DỮ LIỆU VÀ CẤU TRÚC DỮ LIỆU
================================================================================

2.1. THÔNG TIN NGUỒN DỮ LIỆU
------------------------------------------------------------
Dataset: Heart Disease 2020 Cleaned
File dữ liệu: heart_2020_cleaned.csv
URL: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

Mục tiêu bài toán:
- Dự đoán một mẫu dữ liệu có thuộc nhóm mắc bệnh tim hay không.
- Biến mục tiêu là HeartDisease, gồm hai giá trị:
  * No  : không mắc bệnh tim
  * Yes : có mắc bệnh tim

2.2. CẤU TRÚC DỮ LIỆU
------------------------------------------------------------
Tổng quan:
- Số bản ghi ban đầu: 319,795
- Số cột ban đầu: 18
- Số cột đầu vào: 17
- Số cột mục tiêu: 1

Các cột trong bộ dữ liệu:
1. HeartDisease       : Biến mục tiêu, No/Yes
2. BMI                : Chỉ số khối cơ thể
3. Smoking            : Có hút thuốc hay không
4. AlcoholDrinking    : Có uống rượu hay không
5. Stroke             : Tiền sử đột quỵ
6. PhysicalHealth     : Số ngày sức khỏe thể chất không tốt
7. MentalHealth       : Số ngày sức khỏe tinh thần không tốt
8. DiffWalking        : Khó khăn khi đi bộ
9. Sex                : Giới tính
10. AgeCategory       : Nhóm tuổi
11. Race              : Chủng tộc
12. Diabetic          : Tình trạng tiểu đường
13. PhysicalActivity  : Có hoạt động thể chất hay không
14. GenHealth         : Đánh giá sức khỏe tổng quát
15. SleepTime         : Thời gian ngủ
16. Asthma            : Hen suyễn
17. KidneyDisease     : Bệnh thận
18. SkinCancer        : Ung thư da

2.3. NHÓM KIỂU DỮ LIỆU
------------------------------------------------------------
A. Biến số:
- BMI
- PhysicalHealth
- MentalHealth
- SleepTime

B. Biến nhị phân Yes/No:
- Smoking
- AlcoholDrinking
- Stroke
- DiffWalking
- PhysicalActivity
- Asthma
- KidneyDisease
- SkinCancer

C. Biến có thứ tự:
- AgeCategory
- GenHealth

D. Biến danh mục:
- Sex
- Race
- Diabetic

2.4. ĐẶC ĐIỂM QUAN TRỌNG CỦA DỮ LIỆU
------------------------------------------------------------
- Dữ liệu có hiện tượng mất cân bằng lớp.
- Lớp No có số mẫu lớn hơn nhiều so với lớp Yes.
- Vì vậy, khi đánh giá mô hình không chỉ dùng Accuracy.
- Cần quan tâm thêm Precision, Recall, F1-score và ROC-AUC, đặc biệt cho lớp Yes.

================================================================================
PHẦN 3: TỔ CHỨC VÀ CẤU TRÚC CHƯƠNG TRÌNH
================================================================================

3.1. CẤU TRÚC THƯ MỤC DỰ ÁN
------------------------------------------------------------
PROJECTML/
|
|-- Code_ML_Gr8.ipynb                       # Notebook chính của nhóm
|-- heart_2020_cleaned.csv                  # Dữ liệu chính
|-- requirements.txt                        # Danh sách thư viện cần cài đặt
|-- README.md                               # Tài liệu hướng dẫn và mô tả dự án

3.2. CẤU TRÚC NOTEBOOK CHÍNH
------------------------------------------------------------
Notebook Code_ML_Gr8.ipynb gồm các phần chính:

1. Cấu hình chung
   - Import thư viện.
   - Đặt RANDOM_STATE.
   - Đặt đường dẫn dữ liệu.
   - Đặt các tham số cấu hình cho DBSCAN, KNN và trực quan hóa.

2. Tiền xử lý dữ liệu
   - Đọc dữ liệu.
   - Mô tả thông tin ban đầu.
   - Trực quan hóa dữ liệu gốc.
   - Kiểm tra missing values, duplicates và giá trị không hợp lệ.
   - Xử lý trùng lặp và outlier.
   - Chia train/test.
   - Chuẩn hóa cột số.
   - Encode và one-hot các cột còn lại.
   - Mô tả dữ liệu sau chuẩn hóa.

3. Phân tích và trực quan hóa dữ liệu
   - PCA explained variance.
   - Trực quan hóa PC1-PC6.
   - Trực quan hóa mối quan hệ giữa principal components và đầu ra.
   - LDA giảm chiều có giám sát.
   - So sánh PCA và LDA.

4. Phân cụm dữ liệu bằng DBSCAN
   - Sử dụng tối đa 50,000 mẫu dữ liệu sau tiền xử lý.
   - Chuyển dữ liệu về PCA 2D.
   - Tìm eps phù hợp.
   - Chạy DBSCAN.
   - Đánh giá cụm bằng các độ đo định lượng.
   - Trực quan hóa cụm và nhãn.

5. Phân loại
   - Tạo hàm đánh giá chung.
   - Tạo các bộ dữ liệu RAW, PCA và LDA cho từng split.
   - Chạy KNN, Logistic Regression và Linear SVM.
   - Trực quan hóa confusion matrix.
   - Trực quan hóa đường cong ROC-AUC trên dữ liệu RAW với split 80/20.
   - Nhận xét kết quả và overfit.

6. Hồi quy từ score phân lớp
   - Lấy Logistic probability score.
   - Lấy Linear SVM decision score.
   - Tạo dữ liệu đầu vào gốc và PCA 1/3 dimensions.
   - Chạy Linear Regression và Ridge Regression.
   - Đánh giá bằng MAE, RMSE và R2.

7. Kết luận ngắn
   - Tổng hợp các kết quả chính.
   - Nhận xét tính phù hợp của pipeline.

3.3. LUỒNG THỰC THI CHƯƠNG TRÌNH
------------------------------------------------------------
Bước 1: Khởi động và import thư viện
  - Import numpy, pandas, matplotlib, seaborn.
  - Import các module của scikit-learn.
  - Cấu hình tham số chung.

Bước 2: Đọc và mô tả dữ liệu
  - Đọc file heart_2020_cleaned.csv.
  - Hiển thị shape, columns, dtypes.
  - Thống kê phân bố biến mục tiêu.

Bước 3: Trực quan hóa dữ liệu gốc
  - Vẽ phân bố nhãn.
  - Vẽ phân bố các cột số.
  - Vẽ các cột phân loại quan trọng.

Bước 4: Làm sạch dữ liệu
  - Kiểm tra missing values.
  - Xóa duplicates.
  - Kiểm tra outlier.
  - Cap outlier trên các cột số.

Bước 5: Chia train/test
  - Tạo các split 80/20, 70/30 và 60/40.
  - Sử dụng stratify để giữ tỉ lệ nhãn.

Bước 6: Chuẩn hóa và mã hóa
  - Chuẩn hóa cột số bằng StandardScaler.
  - Encode các cột nhị phân và thứ tự.
  - One-hot encoding cho cột danh mục.
  - Đồng bộ cột train/test.

Bước 7: Giảm chiều
  - PCA để đánh giá phương sai được bảo toàn.
  - LDA để đánh giá khả năng tách lớp.

Bước 8: Phân cụm
  - Chạy DBSCAN trên dữ liệu sau khi bỏ trường đầu ra.
  - Đánh giá bằng các chỉ số clustering.
  - Trực quan hóa cụm và nhãn.

Bước 9: Phân loại
  - Chạy KNN.
  - Chạy Logistic Regression.
  - Chạy Linear SVM.
  - So sánh kết quả trên RAW, PCA và LDA.

Bước 10: Hồi quy từ score
  - Dùng score của Logistic Regression và Linear SVM.
  - Chạy Linear Regression và Ridge Regression.
  - So sánh dữ liệu gốc và PCA 1/3 dimensions.

3.4. THƯ VIỆN SỬ DỤNG
------------------------------------------------------------
Core libraries:
- numpy: tính toán số học và mảng.
- pandas: đọc và xử lý dữ liệu bảng.
- matplotlib: vẽ biểu đồ.
- seaborn: trực quan hóa thống kê.

Machine Learning:
- scikit-learn:
  * train_test_split
  * StandardScaler
  * OneHotEncoder
  * OrdinalEncoder
  * PCA
  * LinearDiscriminantAnalysis
  * DBSCAN
  * NearestNeighbors
  * KNeighborsClassifier
  * LogisticRegression
  * LinearSVC
  * LinearRegression
  * Ridge
  * Các metric đánh giá phân loại, phân cụm và hồi quy

================================================================================
PHẦN 4: CÁC KỊCH BẢN THỰC NGHIỆM
================================================================================

4.1. KỊCH BẢN 1: PCA
------------------------------------------------------------
Mục đích:
- Giảm chiều dữ liệu đầu vào.
- Đánh giá lượng thông tin được bảo toàn thông qua explained variance.
- Tìm số principal components cần để giữ lại ít nhất 90% phương sai.

Cách thực hiện:
- Fit PCA trên tập train đã chuẩn hóa và mã hóa.
- Tính explained_variance_ratio_.
- Tính cumulative variance.
- Trực quan hóa PC1-PC6 theo từng cặp.
- Dùng PCA làm một bộ dữ liệu đầu vào cho các mô hình phân loại.

Đầu ra:
- Bảng explained variance.
- Biểu đồ individual variance và cumulative variance.
- Biểu đồ scatter theo các cặp principal components.

4.2. KỊCH BẢN 2: LDA
------------------------------------------------------------
Mục đích:
- Giảm chiều có giám sát dựa trên nhãn HeartDisease.
- Tìm trục LD1 giúp tách hai lớp No và Yes tốt hơn.

Cách thực hiện:
- Fit LinearDiscriminantAnalysis(n_components=1) trên X_train và y_train.
- Biến đổi X_train và X_test sang LD1.
- Thống kê LD1 theo từng lớp.
- Vẽ KDE plot của LD1 theo HeartDisease.

Đầu ra:
- Bảng describe của LD1 theo lớp.
- Biểu đồ phân bố LD1.
- Bộ dữ liệu LDA dùng cho phân loại.

4.3. KỊCH BẢN 3: DBSCAN
------------------------------------------------------------
Mục đích:
- Tìm các nhóm mẫu có cấu trúc gần nhau trong không gian đặc trưng.
- Đánh giá xem các cụm có liên quan đến nhãn HeartDisease hay không.

Thiết lập:
- Dữ liệu đầu vào: các đặc trưng sau tiền xử lý, không gồm HeartDisease.
- Cấu hình hiện tại: DBSCAN_SAMPLE_SIZE = 50000.
- Giảm về 2 chiều bằng PCA để giảm chi phí và phục vụ trực quan hóa.
- min_samples mặc định trong notebook là 20.
- eps được thử nghiệm bằng danh sách giá trị và chọn giá trị phù hợp theo metric.

Metric đánh giá:
- Silhouette Score: càng cao càng tốt.
- Davies-Bouldin Score: càng thấp càng tốt.
- Calinski-Harabasz Score: càng cao càng tốt.
- Adjusted Rand Index: đo tương đồng giữa cụm và nhãn thật.
- Normalized Mutual Information: mức độ thông tin chung giữa cụm và nhãn.
- Purity: tỉ lệ mẫu thuộc lớp đa số trong từng cụm.
- Noise ratio: tỉ lệ điểm bị DBSCAN gán là nhiễu.

4.4. KỊCH BẢN 4: KNN
------------------------------------------------------------
Mô tả:
- KNN là mô hình phân loại dựa trên các điểm láng giềng gần nhất.
- Mô hình không học tham số phức tạp, mà lưu dữ liệu train và dự đoán bằng khoảng cách.

Thiết lập trong notebook:
- KNeighborsClassifier
- n_neighbors = 15
- weights = 'distance'
- metric = 'minkowski'
- p = 2, tương ứng khoảng cách Euclidean

Dữ liệu thử nghiệm:
- RAW
- PCA
- LDA

Tỉ lệ chia:
- 80/20
- 70/30
- 60/40

Lưu ý:
- KNN tốn bộ nhớ và thời gian khi dữ liệu lớn.
- Notebook hiện đặt KNN_TRAIN_SAMPLE_SIZE = 50000 và KNN_TEST_SAMPLE_SIZE = 15000.

4.5. KỊCH BẢN 5: LOGISTIC REGRESSION
------------------------------------------------------------
Mô tả:
- Logistic Regression là mô hình phân loại tuyến tính.
- Mô hình học xác suất một mẫu thuộc lớp Yes.

Thiết lập trong notebook:
- class_weight = 'balanced'
- max_iter = 1000
- C = 1.0
- penalty = 'l2'
- solver = 'lbfgs'

Mục đích:
- Tạo baseline tuyến tính có khả năng xử lý mất cân bằng lớp.
- Lấy predict_proba làm score xác suất cho bài toán hồi quy ở phần sau.

4.6. KỊCH BẢN 6: LINEAR SVM
------------------------------------------------------------
Mô tả:
- Linear SVM là mô hình phân loại tuyến tính.
- Mô hình tìm siêu phẳng phân tách hai lớp với biên độ lớn.
- Đây là soft-margin SVM vì cho phép một số điểm nằm sai phía hoặc nằm trong lề.

Thiết lập trong notebook:
- LinearSVC
- class_weight = 'balanced'
- C = 1.0
- max_iter được tăng để hỗ trợ hội tụ.

Mục đích:
- So sánh với Logistic Regression trong nhóm mô hình tuyến tính.
- Lấy decision_function làm score phân lớp cho bài toán hồi quy.

4.7. KỊCH BẢN 7: HỒI QUY TỪ SCORE PHÂN LỚP
------------------------------------------------------------
Mục đích:
- Chuyển bài toán phân loại thành bài toán hồi quy.
- Biến mục tiêu mới là score sinh ra từ mô hình phân loại.

Score được dùng:
- Logistic probability score: xác suất dự đoán lớp Yes.
- Linear SVM decision score: giá trị hàm quyết định của SVM.

Mô hình hồi quy:
- Linear Regression
- Ridge Regression

Dữ liệu đầu vào:
- Original processed features
- PCA 1/3 dimensions

Metric đánh giá:
- MAE
- RMSE
- R2

================================================================================
PHẦN 5: KẾT QUẢ VÀ ĐÁNH GIÁ
================================================================================

5.1. CÁC CHỈ SỐ ĐÁNH GIÁ PHÂN LOẠI
------------------------------------------------------------
1. Accuracy:
   - Tỉ lệ dự đoán đúng trên tổng số mẫu.
   - Không nên dùng một mình vì dữ liệu bị mất cân bằng lớp.

2. Precision:
   - Trong các mẫu dự đoán là Yes, có bao nhiêu mẫu thật sự là Yes.
   - Quan trọng khi muốn giảm false positive.

3. Recall:
   - Trong các mẫu thật sự là Yes, mô hình phát hiện được bao nhiêu.
   - Quan trọng với bài toán bệnh tim vì bỏ sót ca bệnh có thể nghiêm trọng.

4. F1-score:
   - Trung bình điều hòa của Precision và Recall.
   - Phù hợp hơn Accuracy khi dữ liệu mất cân bằng.

5. ROC-AUC:
   - Đo khả năng phân biệt hai lớp theo các ngưỡng quyết định khác nhau.
   - Giá trị càng gần 1 càng tốt.
   - Notebook có cell vẽ đường cong ROC-AUC cho KNN, Logistic Regression và Linear SVM trên dữ liệu RAW với split 80/20.

6. Overfit Gap Accuracy:
   - Train Accuracy - Test Accuracy.
   - Nếu gap lớn, mô hình có dấu hiệu overfit.

5.2. CÁC CHỈ SỐ ĐÁNH GIÁ PHÂN CỤM
------------------------------------------------------------
1. Silhouette Score:
   - Đo mức độ gần nhau trong cùng cụm và tách xa giữa các cụm.
   - Giá trị càng cao càng tốt.

2. Davies-Bouldin Score:
   - Đo mức độ chồng lấn giữa các cụm.
   - Giá trị càng thấp càng tốt.

3. Calinski-Harabasz Score:
   - So sánh phương sai giữa cụm và trong cụm.
   - Giá trị càng cao càng tốt.

4. Adjusted Rand Index:
   - So sánh nhãn cụm với nhãn thật HeartDisease.
   - Gần 1 là trùng khớp tốt, gần 0 là gần như ngẫu nhiên.

5. Normalized Mutual Information:
   - Đo lượng thông tin chung giữa cụm và nhãn thật.
   - Càng cao càng có liên hệ giữa cụm và nhãn.

6. Cluster Purity:
   - Đo một cụm có bị chi phối bởi một nhãn chính hay không.
   - Purity cao chưa chắc cụm tốt nếu số cụm quá nhiều, nên cần đọc kèm metric khác.

5.3. CÁC CHỈ SỐ ĐÁNH GIÁ HỒI QUY
------------------------------------------------------------
1. MAE:
   - Sai số tuyệt đối trung bình.
   - Càng thấp càng tốt.

2. RMSE:
   - Căn bậc hai của sai số bình phương trung bình.
   - Phạt nặng hơn với lỗi lớn.

3. R2:
   - Mức độ mô hình giải thích được biến thiên của score.
   - Càng gần 1 càng tốt.

5.4. NHẬN XÉT TỔNG QUAN
------------------------------------------------------------
- Dữ liệu HeartDisease bị mất cân bằng lớp, vì vậy Recall và F1-score của lớp Yes
  cần được quan tâm nhiều hơn Accuracy.
- KNN là mô hình cơ bản, dễ hiểu, nhưng tốn chi phí dự đoán cao với dữ liệu lớn.
- Logistic Regression phù hợp làm baseline tuyến tính và dễ giải thích.
- Linear SVM cũng là mô hình tuyến tính, phù hợp khi dữ liệu có nhiều đặc trưng sau mã hóa.
- PCA giúp giảm chiều theo phương sai nhưng không trực tiếp tối ưu cho phân tách nhãn.
- LDA có dùng nhãn nên thường thể hiện rõ hơn mức độ tách hai lớp trên trục LD1.
- DBSCAN giúp kiểm tra cấu trúc cụm tự nhiên của dữ liệu, nhưng cần chọn eps hợp lý và vẫn có thể tốn bộ nhớ với 50,000 mẫu.
- Hồi quy từ score giúp kiểm tra khả năng xấp xỉ hàm quyết định hoặc xác suất của mô hình phân loại.

================================================================================
PHẦN 6: HƯỚNG DẪN SỬ DỤNG
================================================================================

6.1. CÀI ĐẶT MÔI TRƯỜNG
------------------------------------------------------------
Bước 1: Kiểm tra Python
```powershell
python --version
```

Bước 2: Tạo môi trường ảo
```powershell
python -m venv .venv
```

Bước 3: Kích hoạt môi trường ảo trên Windows PowerShell
```powershell
.\.venv\Scripts\Activate.ps1
```

Bước 4: Cài đặt thư viện
```powershell
pip install -r requirements.txt
```

Nếu thiếu thư viện khi chạy notebook, cài thêm:
```powershell
pip install numpy pandas matplotlib seaborn scikit-learn notebook jupyter
```

6.2. KIỂM TRA DỮ LIỆU
------------------------------------------------------------
Đảm bảo file dữ liệu nằm cùng thư mục với notebook:
```text
heart_2020_cleaned.csv
```

6.3. CHẠY NOTEBOOK
------------------------------------------------------------
Cách 1: Jupyter Notebook
```powershell
jupyter notebook
```
Sau đó mở file:
```text
Code_ML_Gr8.ipynb
```

Cách 2: JupyterLab
```powershell
jupyter lab
```

Cách 3: VS Code
- Mở thư mục PROJECTML.
- Mở file Code_ML_Gr8.ipynb.
- Chọn kernel Python phù hợp.
- Chạy từng cell theo thứ tự từ trên xuống dưới hoặc chọn Run All.

6.4. LƯU Ý KHI CHẠY
------------------------------------------------------------
- DBSCAN và KNN có thể tốn bộ nhớ/thời gian, nên notebook đang giới hạn 50,000 mẫu.
- Notebook hiện đặt các biến giới hạn là 50,000 mẫu:
  * DBSCAN_SAMPLE_SIZE = 50000
  * KNN_TRAIN_SAMPLE_SIZE = 50000
  * KNN_TEST_SAMPLE_SIZE = 15000
  * PLOT_SAMPLE_SIZE = 50000
- Nếu laptop yếu hoặc kernel bị treo, có thể đổi các giá trị trên về số cụ thể để lấy mẫu và chạy nhanh hơn.
- Các biểu đồ trong notebook được hiển thị trực tiếp bằng plt.show(); pipeline hiện tại không tự động xuất hoặc lưu hình ảnh ra file.
- Nếu muốn kết quả ổn định hơn, giữ nguyên RANDOM_STATE = 42.
- Nên chạy notebook từ đầu đến cuối, tránh chạy lẻ cell vì nhiều biến được tạo từ các cell trước.

================================================================================
HẾT
================================================================================

Tài liệu được cập nhật: 21/05/2026
