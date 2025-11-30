# KTTT-Project-Nhom-147

## 1. GA bài toán OneMax

* **Tệp:** `GA_original_onemax.py`
* **Mô tả:** Ví dụ kinh điển về GA dùng **mã hóa nhị phân**.
    * **Lai ghép:** Lai ghép một điểm cắt (`crossover`).
    * **Đột biến:** Đột biến lật bit (`mutate`).
    * **Lựa chọn:** Lựa chọn giải đấu (`selection`).

## 2. GA bài toán TSP

* **Tệp:** `GA_original_tsp.py`
* **Mô tả:** GA giải quyết bài toán tối ưu tổ hợp (thứ tự).
    * **Cá thể:** Biểu diễn bằng **thứ tự** các thành phố.
    * **Lai ghép:** Lai ghép Thứ tự (**Ordered Crossover - OX**).
    * **Đột biến:** Đột biến Hoán vị (**Swap Mutation**).
* **Kết quả:** Hình vẽ hiển thị đường đi ban đầu và đường đi tối ưu tìm được.

## 3. RCGA tối ưu giá trị hàm

* **Tệp:** `RCGA.py`
* **Mô tả:** Sử dụng GA để tối ưu các biến **giá trị thực** trên hàm Rastrigin (hàm có nhiều cực trị cục bộ).
    * **Lai ghép:** **Simulated Binary Crossover (SBX)** - rất hiệu quả cho mã hóa thực.
    * **Đột biến:** **Polynomial Mutation**.

## 4. NSGA-II tối ưu đa mục tiêu

* **Tệp:** `NSGA2.py`
* **Mô tả:** Triển khai thuật toán NSGA-II (dùng thư viện `pymoo`) để tìm **Pareto Front** cho bài toán **ZDT3**.
    * NSGA-II là tiêu chuẩn để giải quyết các bài toán có nhiều mục tiêu mâu thuẫn.
* **Kết quả:** Đồ thị hiển thị tập nghiệm tối ưu **(Pareto Front)** tìm được.
