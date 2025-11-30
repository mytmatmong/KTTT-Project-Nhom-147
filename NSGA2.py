# ===================================================================
# NON-DOMINATED SORTING GENETIC ALGORITHM II (NSGA-II)
# Giải bài toán đa mục tiêu ZDT3
# Dùng thư viện pymoo
# ===================================================================

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt

# -------------------- 1. Bài toán ZDT3 --------------------
# 2 mục tiêu mâu thuẫn:
#   f1(x) = x1
#   f2(x) = g(x) * (1 - sqrt(f1/g) - (f1/g)*sin(10*π*f1))
# Với 30 biến x ∈ [0,1], g = 1 + 9*mean(x2..x30)
# → Pareto Front có 5 đoạn rời rạc

problem = get_problem("zdt3")

# -------------------- 2. Cấu hình NSGA-II --------------------
algorithm = NSGA2(
    pop_size=100,
    eliminate_duplicates=True
)

# -------------------- 3. Chạy tối ưu --------------------
print("Đang chạy NSGA-II trên ZDT3...")
res = minimize(problem,
               algorithm,
               ('n_gen', 300),
               seed=42,
               verbose=False)

# -------------------- 4. Vẽ Pareto Front --------------------
plt.figure(figsize=(10, 7))
# Corrected usage of Scatter for plotting
plotter = Scatter(title="Kết quả NSGA-II trên ZDT3",
                  labels=['f1(x)', 'f2(x)'])
plotter.add(res.F, color='red', s=50, alpha=0.8, label='Nghiệm NSGA-II')

# Vẽ đường Pareto lý thuyết (chỉ để so sánh)
ideal = get_problem("zdt3").pareto_front()
plotter.add(ideal, plot_type="line", color='blue', linewidth=2, linestyle='--', label='Pareto Front lý thuyết')

plotter.do()
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Đã tìm được {len(res.F)} nghiệm trên Pareto Front.")
print("Hình ảnh đã được hiển thị.")