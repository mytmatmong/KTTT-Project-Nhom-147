# Bài toán phân bổ công suất truyền thông vô tuyến

import numpy as np
import matplotlib.pyplot as plt
import random

# -------------------- CÀI ĐẶT BÀI TOÁN TRUYỀN THÔNG --------------------
N_USERS = 16                    # Số người dùng
P_TOTAL = 20.0                  # Tổng công suất trạm gốc (Watt)
NOISE_POWER = 1e-6              # Công suất nhiễu
BITS_PER_USER = 16              # Số bit mã hóa công suất mỗi người dùng
CHROM_LENGTH = N_USERS * BITS_PER_USER   # Độ dài nhiễm sắc thể = 256 bit

# Kênh truyền ngẫu nhiên (Rayleigh fading) – cố định seed để tái hiện
np.random.seed(42)
channel_gains = np.random.exponential(scale=1.0, size=N_USERS)   # |h_i|²

# -------------------- HÀM HỖ TRỢ --------------------
def decode_chromosome(chrom):
    """Giải mã chuỗi nhị phân thành vector công suất thực"""
    powers = np.zeros(N_USERS)
    for i in range(N_USERS):
        start = i * BITS_PER_USER
        end = start + BITS_PER_USER
        decimal = int(''.join(map(str, chrom[start:end])), 2)
        powers[i] = decimal / (2**BITS_PER_USER - 1)   # [0, 1]
    total = np.sum(powers)
    if total > P_TOTAL:
        powers = powers / total * P_TOTAL                  # Chuẩn hóa lại
    return powers

def calculate_sum_rate(powers):
    """Tính tổng tốc độ truyền theo công thức Shannon"""
    sinr = powers * channel_gains / NOISE_POWER
    rates = np.log2(1 + sinr)
    return np.sum(rates)

# -------------------- CÁC HÀM GA CỔ ĐIỂN --------------------
def create_individual():
    return [random.randint(0, 1) for _ in range(CHROM_LENGTH)]

def fitness(individual):
    powers = decode_chromosome(individual)
    return calculate_sum_rate(powers)

def tournament_selection(population, k=5):
    candidates = random.sample(population, k)
    return max(candidates, key=fitness)

def one_point_crossover(p1, p2):
    if random.random() < 0.8:  # pc = 0.8
        point = random.randint(1, CHROM_LENGTH-1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2
    return p1.copy(), p2.copy()

def bit_flip_mutation(individual):
    for i in range(CHROM_LENGTH):
        if random.random() < 0.01:    # pm = 0.01
            individual[i] = 1 - individual[i]
    return individual

# -------------------- CHẠY THUẬT TOÁN --------------------
POP_SIZE = 100
GENERATIONS = 200
history = []
best_individual_ever = None
best_fitness_ever = -1

population = [create_individual() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    # Đánh giá
    fitnesses = [fitness(ind) for ind in population]
    best_idx = np.argmax(fitnesses)
    current_best = fitnesses[best_idx]
    
    if current_best > best_fitness_ever:
        best_fitness_ever = current_best
        best_individual_ever = population[best_idx].copy()
    
    history.append(best_fitness_ever)
    
    if gen % 50 == 0 or gen == GENERATIONS-1:
        print(f"Thế hệ {gen:3d} | Sum-rate tốt nhất: {best_fitness_ever:.3f} bps/Hz")

    # Tạo thế hệ mới
    new_pop = [best_individual_ever]  # Elitism
    while len(new_pop) < POP_SIZE:
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        child1, child2 = one_point_crossover(parent1, parent2)
        new_pop.append(bit_flip_mutation(child1))
        if len(new_pop) < POP_SIZE:
            new_pop.append(bit_flip_mutation(child2))
    population = new_pop

# -------------------- KẾT QUẢ CUỐI --------------------
final_powers = decode_chromosome(best_individual_ever)
print("\n=== KẾT QUẢ TỐT NHẤT ===")
print(f"Tổng tốc độ truyền đạt: {best_fitness_ever:.3f} bps/Hz")
print(f"Công suất phân bổ (W): {final_powers.round(3)}")
print(f"Tổng công suất sử dụng: {np.sum(final_powers):.3f} W")

# -------------------- VẼ ĐỒ THỊ --------------------
plt.figure(figsize=(12, 5))

# Hình 1: Hội tụ
plt.subplot(1, 2, 1)
plt.plot(history, color='#2E86C1', linewidth=2.5)
plt.title('Quá trình hội tụ của GA', fontsize=14)
plt.xlabel('Số thế hệ')
plt.ylabel('Tổng tốc độ truyền (bps/Hz)')
plt.grid(True, alpha=0.3)

# Hình 2: Phân bổ công suất cuối cùng
plt.subplot(1, 2, 2)
plt.bar(range(1, N_USERS+1), final_powers, color='#E74C3C', edgecolor='black')
plt.title('Phân bổ công suất tối ưu', fontsize=14)
plt.xlabel('Người dùng')
plt.ylabel('Công suất (W)')
plt.ylim(0, P_TOTAL/2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle('GA CỔ ĐIỂN - PHÂN BỔ CÔNG SUẤT TRUYỀN THÔNG VÔ TUYẾN', 
             fontsize=16, fontweight='bold', y=1.05)

plt.show()
