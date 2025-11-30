# ===================================================================
# REAL-CODED GENETIC ALGORITHM (RCGA)
# Giải bài toán tối ưu hàm Rastrigin (10 chiều)
# ===================================================================

import numpy as np
import matplotlib.pyplot as plt

# -------------------- 1. Bài toán cần giải --------------------
# Hàm Rastrigin - nhiều cực trị cục bộ
# f(x) = 10*n + Σ [xi² - 10*cos(2*π*xi)]
# Không gian: xi ∈ [-5.12, 5.12]
# Tối ưu toàn cục: f(0,0,...,0) = 0

def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# -------------------- 2. Tham số GA --------------------
POP_SIZE = 100        # Kích thước quần thể
DIM = 10              # Số chiều
BOUNDS = (-5.12, 5.12) # Giới hạn biến
GENERATIONS = 300
MUTATION_RATE = 0.2
ETA = 20              # Tham số SBX và Mutation

# -------------------- 3. Các toán tử cho Real-Coded GA --------------------

# Khởi tạo cá thể ngẫu nhiên trong khoảng [low, high]
def create_individual():
    return np.random.uniform(BOUNDS[0], BOUNDS[1], DIM)

# SBX Crossover (Simulated Binary Crossover) - rất mạnh cho biến thực
def sbx_crossover(parent1, parent2):
    if np.random.rand() > 0.9:  # 90% thực hiện crossover
        return parent1.copy(), parent2.copy()
    
    child1, child2 = parent1.copy(), parent2.copy()
    for i in range(DIM):
        if np.random.rand() < 0.5:
            continue
        # Công thức SBX
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (ETA + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (ETA + 1))
            child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
            
            # Đảm bảo không vượt giới hạn
            child1[i] = np.clip(child1[i], BOUNDS[0], BOUNDS[1])
            child2[i] = np.clip(child2[i], BOUNDS[0], BOUNDS[1])
    return child1, child2

# Polynomial Mutation
def polynomial_mutation(individual):
    for i in range(DIM):
        if np.random.rand() < MUTATION_RATE:
            u = np.random.rand()
            if u <= 0.5:
                delta = (2 * u) ** (1.0 / (ETA + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1.0 / (ETA + 1))
            individual[i] += delta * (BOUNDS[1] - BOUNDS[0])
            individual[i] = np.clip(individual[i], BOUNDS[0], BOUNDS[1])
    return individual

# -------------------- 4. Chạy thuật toán --------------------
population = [create_individual() for _ in range(POP_SIZE)]
best_history = []

for gen in range(GENERATIONS):
    # Đánh giá fitness
    fitnesses = [rastrigin(ind) for ind in population]
    best_idx = np.argmin(fitnesses)
    best_history.append(fitnesses[best_idx])
    
    if gen % 50 == 0 or gen == GENERATIONS-1:
        print(f"Thế hệ {gen:3d} | Fitness tốt nhất: {fitnesses[best_idx]:.4f}")
    
    # Tạo thế hệ mới
    new_population = []
    # Elitism: giữ lại 2 cá thể tốt nhất
    elite_idx = np.argsort(fitnesses)[:2]
    new_population.extend([population[i].copy() for i in elite_idx])
    
    while len(new_population) < POP_SIZE:
        # Chọn cha mẹ bằng Tournament
        parent1 = population[np.random.choice(np.argsort(fitnesses)[:50])]
        parent2 = population[np.random.choice(np.argsort(fitnesses)[:50])]
        
        child1, child2 = sbx_crossover(parent1, parent2)
        child1 = polynomial_mutation(child1)
        child2 = polynomial_mutation(child2)
        
        new_population.extend([child1, child2])
    
    population = new_population[:POP_SIZE]

# -------------------- 5. Kết quả --------------------
final_best = population[np.argmin([rastrigin(ind) for ind in population])]
final_fitness = rastrigin(final_best)

print("\n=== KẾT QUẢ CUỐI CÙNG ===")
print(f"Fitness tối ưu: {final_fitness:.6f}")
print(f"Nghiệm tốt nhất: {final_best.round(4)}")

# Vẽ đồ thị hội tụ
plt.figure(figsize=(10, 6))
plt.plot(best_history, color='purple', linewidth=2.5)
plt.title('Hội tụ của RCGA trên hàm Rastrigin', fontsize=14)
plt.xlabel('Thế hệ')
plt.ylabel('Fitness (càng nhỏ càng tốt)')
plt.grid(True, alpha=0.3)
plt.show()