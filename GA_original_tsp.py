import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Bài toán TSP: Tìm đường đi ngắn nhất qua 20 thành phố ---
# Mô tả: Các thành phố được biểu diễn bằng tọa độ (x, y) trong không gian 2D.
# Mục tiêu: Tìm thứ tự ghé thăm các thành phố sao cho tổng quãng đường là nhỏ nhất.

# --- Tạo dữ liệu: 20 thành phố ngẫu nhiên ---
np.random.seed(42)
n_cities = 20
cities = np.random.rand(n_cities, 2) * 100  # tọa độ x, y từ 0-100

# Tính ma trận khoảng cách
def distance_matrix(cities):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.sqrt(((cities[i]-cities[j])**2).sum())
    return dist

dist_matrix = distance_matrix(cities)

# --- Các hàm GA cho TSP ---
def create_individual():
    ind = list(range(n_cities))
    random.shuffle(ind)
    return ind

def fitness(individual):
    total = 0
    for i in range(n_cities):
        total += dist_matrix[individual[i]][individual[(i+1)%n_cities]]
    return total  # Quãng đường càng nhỏ càng tốt

def tournament_selection(population, k=5):
    candidates = random.sample(population, k)
    return min(candidates, key=fitness)

def ordered_crossover(p1, p2):
    start, end = sorted(random.sample(range(n_cities), 2))
    child = [-1] * n_cities
    child[start:end] = p1[start:end]
    remaining = [x for x in p2 if x not in child[start:end]]
    child[:start] = remaining[:start]
    child[end:] = remaining[start:]
    return child

def swap_mutation(individual):
    if random.random() < 0.8:  # 80% có đột biến
        i, j = random.sample(range(n_cities), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# --- Chạy GA ---
POP_SIZE = 200
GENERATIONS = 500

population = [create_individual() for _ in range(POP_SIZE)]
best_history = []

best_ever = None
best_distance = float('inf')

for gen in range(GENERATIONS):
    # Elitism: giữ lựa chọn tốt nhất
    population.sort(key=fitness)
    if fitness(population[0]) < best_distance:
        best_distance = fitness(population[0])
        best_ever = population[0][:]
    
    best_history.append(best_distance)
    
    if gen % 100 == 0 or gen == GENERATIONS-1:
        print(f"Thế hệ {gen:3d} | Quãng đường tốt nhất: {best_distance:.2f}")

    new_pop = [best_ever[:]]  # Elitism
    
    while len(new_pop) < POP_SIZE:
        p1 = tournament_selection(population)
        p2 = tournament_selection(population)
        child = ordered_crossover(p1[:], p2[:])
        child = swap_mutation(child)
        new_pop.append(child)
    
    population = new_pop

plt.figure(figsize=(15, 6))

# Trước tối ưu
plt.subplot(1, 3, 1)
random_route = create_individual()
x = [cities[i][0] for i in random_route] + [cities[random_route[0]][0]]
y = [cities[i][1] for i in random_route] + [cities[random_route[0]][1]]
plt.plot(x, y, 'co-')
plt.title(f"Đường đi ban đầu\nĐộ dài ≈ {fitness(random_route):.1f}")
for i, (xi, yi) in enumerate(cities):
    plt.text(xi+1, yi+1, str(i), fontsize=12)

# Sau tối ưu
plt.subplot(1, 3, 2)
x = [cities[i][0] for i in best_ever] + [cities[best_ever[0]][0]]
y = [cities[i][1] for i in best_ever] + [cities[best_ever[0]][1]]
plt.plot(x, y, 'ro-', linewidth=2.5)
plt.title(f"Đường đi sau GA (500 thế hệ)\nTốt nhất: {best_distance:.2f}")
for i, (xi, yi) in enumerate(cities):
    plt.text(xi+1, yi+1, str(i), fontsize=12)

# Đồ thị hội tụ
plt.subplot(1, 3, 3)
plt.plot(best_history, color='green', linewidth=2)
plt.xlabel('Thế hệ')
plt.ylabel('Quãng đường ngắn nhất')
plt.title('Hội tụ của GA trên TSP 20 thành phố')
plt.grid(True, alpha=0.3)

plt.suptitle('THUẬT TOÁN DI TRUYỀN GIẢI BÀI TOÁN NGƯỜI DU LỊCH (TSP)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()