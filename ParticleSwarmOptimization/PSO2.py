import random

c1 = 1.5
c2 = 1.5

dimension = 20
iterations = 3000

velocity = []
pos = []
pBest = []
gBest = 0

time_value = []
accuracy_value = []


def fitness(x):
    return -(x * x) + 5


def updatePosition():
    for i in range(dimension):
        pos[i] = pos[i] + velocity[i]


def updateVelocities():
    for i in range(dimension):
        r1 = random.random()
        r2 = random.random()
        velocity[i] = (velocity[i]) + (c1 * r1 * (gBest - pos[i])) + (c2 * r2 * (pBest[i] - pos[i]))


for i in range(dimension):
    pos.append(random.randint(0, 1))
    velocity.append(random.randint(0, 1))
    pBest.append(pos[i])

for i in range(iterations):
    for i in range(dimension):
        if fitness(pos[i]) > fitness(pBest[i]):
            pBest[i] = pos[i]
    gBest = fitness(max(pBest))
    updateVelocities()
    updatePosition()

print("output PSO : ", gBest)