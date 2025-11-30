# GA OneMax 100 bit
import random
import matplotlib.pyplot as plt

# --- 1. THAM SỐ CẤU HÌNH ---
POP_SIZE = 100           # Kích thước quần thể
CHROM_LENGTH = 100       # Độ dài nhiễm sắc thể
GENERATIONS = 100        # Số thế hệ tối đa
MUTATION_RATE = 0.01     # Tỷ lệ đột biến
CROSSOVER_RATE = 0.8     # Tỷ lệ lai ghép
TOURNAMENT_SIZE = 5      # Kích thước nhóm đấu

# --- 2. KHỞI TẠO VÀ ĐÁNH GIÁ ---

def create_individual():
    # Tạo cá thể ngẫu nhiên (chuỗi nhị phân)
    return [random.randint(0, 1) for _ in range(CHROM_LENGTH)]

def fitness(individual):
    # Hàm mục tiêu: Tổng số bit '1'
    return sum(individual)

# --- 3. LỰA CHỌN ---

def selection(population):
    # Lựa chọn giải đấu
    tournament_candidates = random.sample(population, TOURNAMENT_SIZE)
    return max(tournament_candidates, key=fitness)

# --- 4. LAI GHÉP ---

def crossover(p1, p2):
    # Lai ghép một điểm cắt
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, CHROM_LENGTH - 1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2
    return p1, p2

# --- 5. ĐỘT BIẾN ---

def mutate(ind):
    # Đột biến bit flip
    for i in range(len(ind)):
        if random.random() < MUTATION_RATE:
            ind[i] = 1 - ind[i]
    return ind

# --- 6. VÒNG LẶP CHÍNH ---
population = [create_individual() for _ in range(POP_SIZE)]
best_history = []

for gen in range(GENERATIONS):
    # Đánh giá và lưu lại fitness tốt nhất
    population.sort(key=fitness, reverse=True)
    best_fitness = fitness(population[0])
    best_history.append(best_fitness)

    if gen % 25 == 0 or gen == GENERATIONS - 1:
        print(f"Thế hệ {gen}: Fitness tốt nhất = {best_fitness}/{CHROM_LENGTH}")

    # Tạo quần thể mới
    new_pop = [population[0]]  # Elitism
    while len(new_pop) < POP_SIZE:
        p1 = selection(population)
        p2 = selection(population)
        c1, c2 = crossover(p1[:], p2[:])
        new_pop.append(mutate(c1))
        new_pop.append(mutate(c2))
    population = new_pop[:POP_SIZE]

# --- 7. HIỂN THỊ KẾT QUẢ ---
print(f"\nKết quả cuối: Fitness tốt nhất = {best_history[-1]}/{CHROM_LENGTH}")

plt.figure(figsize=(10, 6))
plt.plot(best_history, color='darkgreen', linewidth=2.5, label='Fitness tốt nhất')
plt.xlabel('Thế hệ', fontsize=12)
plt.ylabel('Fitness', fontsize=12)
plt.title(f'Hội tụ GA - OneMax ({CHROM_LENGTH} bits)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.show()
