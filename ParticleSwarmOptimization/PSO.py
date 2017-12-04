import random

positions = []
velocities = []

for i in range(10):
    positions.append(random.randint(0,1))
    velocities.append(random.randint(0, 1))


def fitness(val):
    return 10 - (val*val)

def evaluation_funtion(positions):
    fit_vals = []
    for pos in positions:
        func_val = 10 - (pos*pos)
        fit_vals.append(func_val)
    return fit_vals


def get_local_best(local_best, new_fit):
    for i in range(len(local_best)):
        if new_fit[i] >= local_best[i]:
            local_best[i] = new_fit[i]

    return local_best


def get_global_best(local_best, global_best, global_best_index):
    for i in range(len(local_best)):
        if fitness(local_best[i]) >= fitness(global_best):
            global_best = local_best[i]
            global_best_index = i
    return global_best_index, global_best


def update(positions, velocities, local_best, global_best):
    for i in range(len(positions)):
        velocities[i] = velocities[i] + 1.5 * random.randint(0,1) * (positions[i] - local_best[i]) + 1.5 * random.randint(0,1) * (positions[i] - global_best)
        positions[i] = positions[i] + velocities[i]

    return positions, velocities

local_best = evaluation_funtion(positions)
global_best = max(local_best)
global_best_index = -1

for i in range(20):
    local_best = get_local_best(local_best, positions)
    global_best_index, global_best = get_global_best(local_best, global_best, global_best_index)
    update(positions, velocities, local_best, global_best)

local_best = get_local_best(local_best, positions)
global_best_index, global_best = get_global_best(local_best, global_best, global_best_index)

global_best = max(local_best)

print(local_best)
print(global_best)