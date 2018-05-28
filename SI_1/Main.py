import numpy as np
import random
import pylab
import time
np.set_printoptions(threshold=np.nan)

#flowMatrix = np.array([[0, 3, 0, 2], [3, 0, 0, 1], [0, 0, 0, 4], [2, 1, 4, 0]])
#distanceMatrix = np.array([[0, 22, 53, 53], [22, 0, 40, 62], [53, 40, 0, 55], [53, 62, 55, 0]])

# A B C D E <-  LOKALIZACJE
#-----------
# 1 2 3 4 5 <- FABRYKI

def initializePopulation(howMany, nrOfFacilities):
    population = np.zeros([howMany, nrOfFacilities], dtype=int)
    facilities = np.arange(1, nrOfFacilities+1)
    for i in range(howMany):
       population[i] = np.random.permutation(facilities)
    return population

def costOfOne(candidate, flowMatrix, distanceMatrix):
    costOfCandidate = 0
    for j in range(0, flowMatrix.shape[1]):
        for k in range(j + 1, flowMatrix.shape[1]):
            currentFacility = candidate[j] - 1
            nextFacility = candidate[k] - 1
            costOfCandidate += distanceMatrix[j][k] * flowMatrix[currentFacility][nextFacility]
    return costOfCandidate

def countCost(population, flowMatrix, distanceMatrix):
    costOfAll = []
    for i in range(0, population.shape[0]):
        costOfAll.append(costOfOne(population[i], flowMatrix, distanceMatrix))
    return costOfAll


def selectParents(population, howMany):
    return population[np.random.choice(population.shape[0], howMany, replace=False), :]

def tournamentSelection(population, nrOfCandidatesInCompetition, costOfAll):
    endOfArray = 0
    nrOfWinners = int(np.ceil(population.shape[0] / nrOfCandidatesInCompetition))
    winners = np.zeros([nrOfWinners, population.shape[1]], dtype=int)
    win = []
    for i in range(0, population.shape[0], nrOfCandidatesInCompetition):
        smallestCostInCompetition = costOfAll[i]
        winner = i
        for j in range (i, i+nrOfCandidatesInCompetition):
            if(costOfAll[j]<smallestCostInCompetition ):
                winner = j
                smallestCostInCompetition = costOfAll[j]
        winners[endOfArray][:] = population[winner]
        win.append(costOfAll[winner])
        endOfArray+=1
    return winners

def roulletteSelection (population, costOfAll, nrOfParents):
    populationSize = population.shape[0]
    minPlusMax = np.min(costOfAll) + np.max(costOfAll)
    reversedCost = (costOfAll-minPlusMax)*(-1)
    sumOfReversedCost = np.sum(reversedCost)
    probabilities = reversedCost/sumOfReversedCost
    winnersIndexes = np.random.choice(populationSize, nrOfParents,p=probabilities)
    winners = population[winnersIndexes]
    return winners

def PMX(parent1, parent2):
    """Executes a partially matched crossover (PMX) on the input individuals.
    :param parent1: The first individual participating in the crossover.
    :param parent2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    """
    parent1 = list(np.asarray(parent1) - 1)
    parent2 = list(np.asarray(parent2) - 1)
    size = min(len(parent1), len(parent2))
    p1, p2 = [0] * size, [0] * size
    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[parent1[i]] = i
        p2[parent2[i]] = i
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = parent1[i]
        temp2 = parent2[i]
        # Swap the matched value
        parent1[i], parent1[p1[temp2]] = temp2, temp1
        parent2[i], parent2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    parent1 = list(np.asarray(parent1) + 1)
    parent2 = list(np.asarray(parent2) + 1)
    return parent1, parent2

def onePointCrossover(population):
    splitPoint = int(np.floor(population.shape[1]/2))
    endOfArray = 0
    nrOfPopulation = population.shape[0]
    children = np.zeros([nrOfPopulation*2, population.shape[1]], dtype=int)
    win = []
    for i in range(0, nrOfPopulation, 2):
        firstParentGen = population[i][:splitPoint]
        secondParentGen = population[i+1][splitPoint:]
        firstParent2Gen = population[nrOfPopulation-i-1][:splitPoint]
        secondParent2Gen = population[nrOfPopulation-i-2][splitPoint:]

        children[2 * i][:] = np.concatenate([firstParentGen, secondParentGen])
        children[2 * i + 1][:] = np.concatenate([secondParentGen, firstParentGen])
        children[2 * i + 2][:] = np.concatenate([firstParent2Gen, secondParent2Gen])
        children[2 * i + 3][:] = np.concatenate([secondParent2Gen, firstParent2Gen])

    return children

def crossover(population, nrOfChildren, crossoverProbability, costOfParents):
    nrOfPopulation = population.shape[0]
    nrOfFacilities = population.shape[1]
    notCrossedObjectsNr = int(nrOfPopulation * (1 - crossoverProbability))
    children = np.zeros([nrOfChildren, nrOfFacilities], dtype=int)
    sortedCostArgs = np.argsort(costOfParents)
    for i in range (0, notCrossedObjectsNr):
        children[i] = population[sortedCostArgs[i]]
    for i in range(notCrossedObjectsNr, nrOfChildren, 2):
        if(i<nrOfChildren):
            firstParent = population[np.random.randint(nrOfPopulation)][:]
            secondParent = population[np.random.randint(nrOfPopulation)][:]
            if (np.array_equal(firstParent, secondParent)):
                firstParent = population[np.random.randint(nrOfPopulation)][:]
                secondParent = population[np.random.randint(nrOfPopulation)][:]

            child1, child2 = PMX(firstParent, secondParent)
            """
            if(random.random() < crossoverProbability):
                child1, child2 = PMX(firstParent, secondParent)

            else:
                child1 = firstParent
                child2 = secondParent
            """
            children[i][:] = child1
            if(i+1 <nrOfChildren):
                children[i + 1][:] = child2


    return children

def mutate(object):
    nrOfGenes = len(object)
    a = np.random.randint(0, nrOfGenes)
    b = np.random.randint(0, nrOfGenes)
    while (a == b):
        a = np.random.randint(0, nrOfGenes)
        b = np.random.randint(0, nrOfGenes)
    gen1 = object[a]
    gen2 = object[b]
    object[a] = gen2
    object[b] = gen1
    return object

def mutation(population, pm):
    populationSize = population.shape[0]
    nrOfMutations = int(np.ceil(populationSize*pm))
    for i in range (nrOfMutations):
        index = np.random.randint(0, populationSize)
        population[index] = mutate(population[index])
    return population

def evaluate(population, maxNrOfGeneration, flowMatrix, distanceMatrix, px, pm, tour, tm, generation=0):
    start = time.time()
    timeout = start + tm
    popNr = population.shape[1]
    generations = []
    averageValues = []
    minValues = []
    maxValues = []
    costOfAll = countCost(population, flowMatrix, distanceMatrix)
    minCost, minIndex, avgCost, maxCost = np.min(costOfAll), np.argmin(costOfAll), np.mean(costOfAll), np.max(costOfAll)
    #while(generation < maxNrOfGeneration ):
    while(timeout> time.time()):
            #parents = roulletteSelection(population, costOfAll, 20)
            parents = tournamentSelection(population, tour, costOfAll)
            costOfParents = countCost(parents, flowMatrix, distanceMatrix)
            children = crossover(parents, population.shape[0], px, costOfParents)
            population = mutation(children, pm)
            costOfAll = countCost(population, flowMatrix, distanceMatrix)
            minCost, minIndex, avgCost, maxCost = np.min(costOfAll), np.argmin(costOfAll), np.mean(
                costOfAll), np.max(costOfAll)
            generations.append(generation)
            averageValues.append(avgCost)
            minValues.append(minCost)
            maxValues.append(maxCost)
            generation += 1
    end = time.time()
    """
    print('Duration: ', end-start)
    print(generations)
    print(averageValues)
    pylab.plot(generations, averageValues, label="Average")
    pylab.plot(generations, minValues, label="Min")
    pylab.plot(generations, maxValues, label="Max")
    pylab.legend()
    pylab.show()
    """
    return np.mean(averageValues)

def randomSearch(populationSize, nrOfFacilities, flowMatrix, distanceMatrix, tm):
    facilities = np.arange(1, nrOfFacilities + 1)
    timeout = time.time() + tm
    minCost = costOfOne(facilities, flowMatrix, distanceMatrix)
    while(timeout > time.time()):
        population = initializePopulation(populationSize, nrOfFacilities)
        costOfAll = countCost(population, flowMatrix, distanceMatrix)
        minPopCost = np.min(costOfAll)
        if(minPopCost< minCost):
            minCost = minPopCost
    return minCost

def greedy(nrOfFacilities, flowMatrix, distanceMatrix, tm):
    facilities = np.arange(1, nrOfFacilities + 1)
    timeout = time.time() + tm
    minCost = costOfOne(facilities, flowMatrix, distanceMatrix )
    while(timeout > time.time()):
        object = np.random.permutation(facilities)
        objectCost = costOfOne(object, flowMatrix, distanceMatrix)
        if(objectCost < minCost):
            minCost = objectCost
        facilities = object
    return minCost

def greedy2(nrOfFacilities, flowMatrix, distanceMatrix):
    facilities = np.zeros(nrOfFacilities, dtype=int)
    facilities[0] = np.random.randint(1, nrOfFacilities+1)
    aviable = np.arange(1, nrOfFacilities+1).tolist()

    aviable.remove(facilities[0])
    for i in range (1, nrOfFacilities):
        facilities[i] = aviable[0]
        minCost = distanceMatrix[i-1][i]*flowMatrix[facilities[i-1]-1][aviable[0]-1]
        tempWinner = aviable[0]
        aviable.remove(tempWinner)
        for j in range(len(aviable)):
            cost = distanceMatrix[i-1][i]*flowMatrix[facilities[i-1]-1][aviable[j]-1]
            if(cost < minCost):
                facilities[i] = aviable[j]
                aviable.remove(facilities[i])
                aviable.append(tempWinner)
                tempWinner = facilities[i]
                minCost = cost

    return costOfOne(facilities, flowMatrix, distanceMatrix)

def greedyTime(nrOfFacilities, flowMatrix, distanceMatrix, tm):
    timeout = time.time() + tm
    minCost = greedy2(nrOfFacilities, flowMatrix, distanceMatrix)
    while(timeout > time.time()):
        cost = greedy2(nrOfFacilities, flowMatrix, distanceMatrix)
        if(cost<minCost):
            minCost = cost
    return minCost
def go (nr, px, pm, popSize, maxGen, tm):
    distanceMatrix = np.loadtxt(str(nr)+'_distance.out', delimiter='  ', dtype=int).reshape(nr, nr)
    flowMatrix = np.loadtxt(str(nr)+'_flow.out', delimiter='  ', dtype=int).reshape(nr, nr)
    population = initializePopulation(popSize, nr)
    return evaluate(population, maxGen, flowMatrix, distanceMatrix, px, pm, 5, tm)



#go(16, 0.0, 0.05, 100, 100)
distanceMatrix16 = np.loadtxt('16_distance.out', delimiter='  ', dtype=int).reshape(16, 16)
flowMatrix16 = np.loadtxt('16_flow.out', delimiter='  ', dtype=int).reshape(16, 16)

#print(greedy2(16, flowMatrix16,distanceMatrix16))

print(greedyTime(16, flowMatrix16, distanceMatrix16, 0.01))

"""

steps = []
gr = []
ran = []
ga = []
for i in range (1, 17, 5):
    steps.append(i)
    gr.append(greedyTime(16, flowMatrix16, distanceMatrix16, i))
    ran.append(randomSearch(100, 16, flowMatrix16, distanceMatrix16, i))
    ga.append(go(16, 0.7, 0.01, 100, 100, i))


pylab.plot(steps, gr, label="Greedy")
pylab.plot(steps, ran, label="Random search")
pylab.plot(steps, ga, label="GA")
pylab.legend()
pylab.show()



steps = []
values = []

for j in range (1, 30, 2):
    temps = []
    steps.append(j/100)
    for i in range (0, 10):
        temps.append(go(16, 0.7, j/100, 100, 100))
    values.append(np.mean(temps))

pylab.plot(steps, values)
pylab.legend()
pylab.show()
"""

"""
4,2
6,7
pylab.plot(steps, values)
pylab.legend()
pylab.show()


pylab.plot(generations, trn, label="Tournament selection")
pylab.plot(generations, rl, label="Roulette selection")
pylab.legend()
pylab.show()



#print(roulletteSelection(population16, countCost(population16, flowMatrix16, distanceMatrix16), 50))
#(9,4,16,1,7,8,6,14,15,11,12,10,5,3,2,13)
#print(costOfOne([9,4,16,1,7,8,6,14,15,11,12,10,5,3,2,13], flowMatrix16, distanceMatrix16))

distanceMatrix16 = np.loadtxt('16_distance.out', delimiter='  ', dtype=int).reshape(16, 16)
flowMatrix16 = np.loadtxt('16_flow.out', delimiter='  ', dtype=int).reshape(16, 16)
population16 = initializePopulation(100, 16)

#print(countCost(initializePopulation(30, 4), flowMatrix, distanceMatrix))
flowMatrix = np.array([[0, 3, 0, 2], [3, 0, 0, 1], [0, 0, 0, 4], [2, 1, 4, 0]])
distanceMatrix = np.array([[0, 22, 53, 53], [22, 0, 40, 62], [53, 40, 0, 55], [53, 62, 55, 0]])

population = initializePopulation(9, 4)
print(population)
costOfAll = countCost(population, flowMatrix, distanceMatrix)
print(population)
print('cost', costOfAll)
parents = tournamentSelection(population, 2, costOfAll)
print(parents)
#print(costOfAll)
#print('cross')
#print(crossover(parents, 50))
#print(PMX([1,2,3,4,5,6,7], [7, 6, 5, 4, 3, 2, 1]))

#print(evaluate(population, 400, flowMatrix, distanceMatrix, 10))


"""

"""
flowMatrix8 = np.loadtxt('8_distance.out', delimiter='  ', dtype=int).reshape(8, 8)
distanceMatrix8 = np.loadtxt('8_flow1.out', delimiter=' ', dtype=int).reshape(8, 8)
population8 = initializePopulation(100, 8)
# 6 5 4 3 7 8 1 3 2
# (array([8, 1, 3, 2, 7, 6, 5, 4]), 904, 3026)
print(evaluate(population8, 920, flowMatrix8, distanceMatrix8, 10))


flowMatrix12 = np.loadtxt('12_distance.out', delimiter='  ', dtype=int).reshape(12, 12)
distanceMatrix12 = np.loadtxt('12_flow.out', delimiter='  ', dtype=int).reshape(12, 12)
population12 = initializePopulation(100, 12)

print(evaluate(population12, 840, flowMatrix12, distanceMatrix12, 10))
#3,10,11,2,12,5,6,7,8,1,4,9)

"""