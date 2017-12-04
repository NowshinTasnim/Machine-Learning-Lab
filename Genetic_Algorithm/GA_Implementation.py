import random
import math
import copy

random.randint(0,1)

chromosome = []
save_current_best = -99999999999
save_best = -9999999999

def initialize_population():
    for i in range(4):
        array = []
        for j in range(6):
            array.append(random.randint(0,1))
        chromosome.append(array)


def fitness(single_chrom):
    val = 0
    flag = 0
    chrom_val = 0
    for i in range(len(single_chrom)):
        if i == 0:
            if single_chrom[i] == 1:
                flag = 1
        else:
            val += math.pow(2,(len(single_chrom)-i-1)) * single_chrom[i]
    if flag == 1:
        val = -val

    chrom_val = val
    val = -(val*val) + 5
    return val, chrom_val



def calculate_fitness(chrom):
    new_fit = []
    chrom_val_list = []
    for i in range(len(chrom)):
        val, chrom_val = fitness(chrom[i])
        new_fit.append(val)
        chrom_val_list.append(chrom_val)
    return new_fit, chrom_val_list


def selection(fit_val):
    first_best = -999999999
    first_index = -1
    second_best = -999999999
    second_index = -1
    for i in range(len(fit_val)):
        if fit_val[i] >= first_best:
            first_best = fit_val[i]
            first_index = i

    for i in range(len(fit_val)):
        if i!=first_index and fit_val[i] >= second_best:
            second_best = fit_val[i]
            second_index = i

    return first_index, second_index

def crossover(first, second):
    mutation_point = random.randint(1,len(first))
    for i in range(mutation_point,len(first)):
        first[i], second[i] = second[i], first[i]

def mutation(chromosome):
    r = random.randint(0,50)
    if (r == 20):
        i = random.randint(0,len(chromosome)-1)
        j = random.randint(0,len(chromosome[i])-1)
        chromosome[i][j] = 1 - chromosome[i][j]

initialize_population()
fit_values, chrom_val = calculate_fitness(chromosome)
for i in range(len(fit_values)):
    if fit_values[i] >= save_current_best:
        save_current_best = fit_values[i]


for iteration in range(1000):
    new_chrom = copy.copy(chromosome)
    frst, sec = selection(fit_values)
    crossover(new_chrom[frst], new_chrom[sec])
    mutation(new_chrom)
    new_fit_values, chrom_val = calculate_fitness(new_chrom)
    print("Iteration " + str(iteration+1))
    print("Binary Chromosome:")
    print(*new_chrom)
    print("Decimal Value: ")
    print(*chrom_val)
    print("Fit Value:")
    print(*new_fit_values)
    print("Best value:")
    print(save_best)
    print("\n")
    for i in range(len(new_fit_values)):
        if new_fit_values[i] >= save_best:
            save_best = new_fit_values[i]

    if save_best >= save_current_best:
        save_current_best = save_best

    chromosome = new_chrom

print(chromosome)
print(save_current_best)