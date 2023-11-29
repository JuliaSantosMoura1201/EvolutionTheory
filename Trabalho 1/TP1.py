import csv
import math
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

class Struct:
    pass

class History:
    def __init__(
        self,
        fit = None,
        sol = None,
        fea = None,
        vio = None
    ):
        self.fit = [] if fit is None else fit
        self.sol = [] if sol is None else sol
        self.fea = [] if fea is None else fea
        self.vio = [] if vio is None else vio

class ProblemDefinition:
    clientsMinCoverage = 0.95
    paMaxCapacity = 54
    paMaxCoverageRadius = 70
    maxAmountOfPAs = 25
    numberOfPlacesToInstallPas = 400/5
    epsilon = 11900

    def __init__(
        self,
        clients = None,
        PAs = None
    ):
        self.clients = clients
        self.PAs = PAs

class Solution:
    def __init__(
        self,
        fitness = 0,
        violation = 0,
        feasible = True,
        currentSolution = None,
        singleObjectiveValue = 0,
        secondObjectiveFitness = 0
    ):
        self.fitness = fitness
        self.violation = violation
        self.feasible = feasible
        self.currentSolution = [] if currentSolution is None else currentSolution
        self.singleObjectiveValue = 0
        self.secondObjectiveFitness = 0
        self.loadStandardDesviation = 0
        self.pasDistanceStandardDesviation = 0

class Client:
    def __init__(self, x, y, band_width):
        self.x = x
        self.y = y
        self.band_width = band_width

class PA:
    def __init__(
        self,
        x,
        y,
        capacity = 0,
        coverage_radius = 0,
        clients = None,
        enabled = False
    ):
        self.x = x
        self.y = y
        self.capacity = capacity
        self.coverage_radius = coverage_radius
        self.clients = [] if clients is None else clients
        self.enabled = enabled

    def updateClientsList(self, client):
        self.capacity += client.band_width
        self.clients.append(client)
        self.enabled = not self.clients

    def _getClientWithBiggestDistance(self, clients):
        if not clients:
            return 0
        return sorted(clients, key = lambda client: getDistanceBetweenPAAndClient(self, client))[-1]

    def calculateCapacity(clients):
        capacity = 0
        for client in clients:
            capacity += client.band_width
        return capacity

    def withNewClients(self, *clients):
        if not clients:
            return self
        newClients = self.clients + list(*clients)
        return PA(
            x = self.x,
            y = self.y,
            coverage_radius = self._getClientWithBiggestDistance(newClients),
            capacity = self.capacity + sum([client.band_width for client in newClients]),
            clients = newClients,
            enabled = True
        )

def getClients():
    clients = []
    with open('clientes.csv', mode='r') as clientes_csv:

        leitor_csv = csv.reader(clientes_csv)

        for linha in leitor_csv:
            newClient = Client(float(linha[0]), float(linha[1]), float(linha[2]))
            clients.append(newClient)
    return clients

def factoryPAs():
    PAs = []
    for x in range(0, 400, 5):
        for y in range(0, 400, 5):
            newPA = PA(x, y)
            PAs.append(newPA)
    return PAs

def filterPasWithClients(PAs):
    return list(filter(lambda PA: len(PA.clients) != 0, PAs))

def filterPAsEnabled(PAs):
    return list(filter(lambda PA: PA.capacity != 0, PAs))

def factoryInitialSolutionToMinimizePAsAmount(problemDefinition):
    solution = Solution()
    initialPAsList = minimizeMultiObjectiveHeuristic(problemDefinition)
    solution.currentSolution = filterPAsEnabled(initialPAsList)
    return solution

def calculateViolation(solution, problemDefinition):
    R1 = percentageOfClientsNotAttendedBellowLimit(solution.currentSolution, problemDefinition)
    #print("percentageOfClientsNotAttended", R1)
    R2 = amountOfPAsAboveCapacity(solution.currentSolution, problemDefinition)
    #print("amountOfPAsAboveCapacity", R2)
    R3 = amountOfPAsClientOutsideMaxRange(solution.currentSolution, problemDefinition)
    #print("amountOfPAsClientOutsideMaxRange", R3)
    R4 = amountOfReplicatedClients(solution.currentSolution)
    #print("amountOfReplicatedClients", R4)
    R5 = amountOfPAsAboveMaxAmount(solution.currentSolution, problemDefinition)
    #print("amountOfPAsAboveMaxAmount", R5)
    return 150 * (R1 + R2 + R3 + R4 + R5)

def objectiveFuntionMinimizeAmountOfPAs(solution, problemDefinition):
    newSolution = copy.deepcopy(solution)
    enabledPAs = filterPAsEnabled(newSolution.currentSolution)

    newSolution.fitness = len(enabledPAs)
    newSolution.secondObjectiveFitness = getTotalDistanceSumBetweenPAsAndClients(enabledPAs)

    newSolution.violation = calculateViolation(solution, problemDefinition)

    newSolution.feasible = newSolution.violation == 0

    enabledPAsLoad = [pa.capacity for pa in enabledPAs]
    newSolution.loadStandardDesviation = np.std(enabledPAsLoad)

    enabledPAsDistance = [sum(getDistanceBetweenPAAndClient(pa, client) for client in pa.clients) for pa in enabledPAs]
    newSolution.pasDistanceStandardDesviation = np.std(enabledPAsDistance)

    return newSolution

def pw(solution, problemDefinition, weights):
    newSolution = objectiveFuntionMinimizeAmountOfPAs(solution, problemDefinition)
    newSolution.singleObjectiveValue = weights[0] * newSolution.fitness + weights[1] * newSolution.secondObjectiveFitness
    return newSolution

# weights n é usado, mas queremos manter a mesma assinatura para pe e pw
def pe(solution, problemDefinition, weights):
    newSolution = objectiveFuntionMinimizeAmountOfPAs(solution, problemDefinition)
    violation = newSolution.secondObjectiveFitness - problemDefinition.epsilon
    violation = violation if(violation > 0) else 0
    newSolution.violation = newSolution.violation + violation
    newSolution.singleObjectiveValue = newSolution.fitness + newSolution.violation
    return newSolution

#R1
def percentageOfClientsNotAttendedBellowLimit(PAs, problemDefinition):
    selectedClients = set()

    for pa in PAs:
        selectedClients.update(pa.clients)

    percentageOfClientsAttended = len(selectedClients)/len(problemDefinition.clients)
    percentageOfClientsNotAttendedBellowLimit = problemDefinition.clientsMinCoverage - percentageOfClientsAttended

    if(percentageOfClientsNotAttendedBellowLimit <= 0):
        return 0
    return percentageOfClientsNotAttendedBellowLimit * 100

#R2
def amountOfPAsAboveCapacity(PAs, problemDefinition):
    amountOfPAsAboveCapacity = 0
    for pa in PAs:
        if pa.capacity > problemDefinition.paMaxCapacity :
            amountOfPAsAboveCapacity += 1
    return amountOfPAsAboveCapacity

#R3
def amountOfPAsClientOutsideMaxRange(PAs, problemDefinition):
    amountOfPAsClientOutsideMaxRange = 0
    for pa in PAs:
        for client in pa.clients:
            if getDistanceBetweenPAAndClient(pa, client) > problemDefinition.paMaxCoverageRadius:
                amountOfPAsClientOutsideMaxRange += 1
    return amountOfPAsClientOutsideMaxRange

#R4
def amountOfReplicatedClients(PAs):
    selectedClients = []

    for pa in PAs:
        selectedClients += pa.clients

    amountOfReplicatedClients = 0
    for client in selectedClients:
        count = selectedClients.count(client)
        if(count > 1):
            amountOfReplicatedClients += count - 1 # subtrai um pq tem que ter ao menos um mesmo

    return amountOfReplicatedClients

#R5
def amountOfPAsAboveMaxAmount(PAs, problemDefinition):
    amountOfPas = len(PAs)
    if amountOfPas <= problemDefinition.maxAmountOfPAs:
        return 0
    return amountOfPas - problemDefinition.maxAmountOfPAs

def minimizeMultiObjectiveHeuristic(problemDefinition):

    def sortPAsForCloseClientsCount(PAs, clients, problemDefinition):
            return sorted(PAs, key = lambda pa: len(selectClientsByDistance(pa, clients, problemDefinition)[0]), reverse = True)

    finalPAsList = []
    candidatePAs = sortPAsForCloseClientsCount(problemDefinition.PAs, problemDefinition.clients, problemDefinition)
    unnalocatedClients = [*problemDefinition.clients]
    while len(unnalocatedClients):
        # Pega o primeiro PA pq ele sempre é o que tem mais clientes no raio
        currentPa = candidatePAs.pop(0)
        # Aloca o máximo de clientes possíveis no PA atual
        finalPAsList, unnalocatedClients = allocateClientesToPAs((finalPAsList, unnalocatedClients), currentPa, problemDefinition)
        # Reordena os PAs em função da quantidade de clientes no raio
        candidatePAs = sortPAsForCloseClientsCount(candidatePAs, unnalocatedClients, problemDefinition)
    return finalPAsList + candidatePAs

def allocateClientesToPAs(PAsAndClients, currentPA, problemDefinition):
    PAs, unnalocatedClients = PAsAndClients

    if not unnalocatedClients:
            return ([*PAs, currentPA], [])

    candidateClients, tooFarClients = selectClientsByDistance(currentPA, unnalocatedClients, problemDefinition)
    selectedClients, restClients = selectClientsUntilCapacity(currentPA, candidateClients, problemDefinition)
    #selectedClients, restClients = selectClientsUntilMaxAmount(currentPA, candidateClients, problemDefinition)

    newPA = currentPA.withNewClients(selectedClients)

    return ([*PAs, newPA], restClients + tooFarClients)

def selectClientsByDistance(PA, clients, problemDefinition):
    selected, unselected = [], []

    for client in clients:
        if getDistanceBetweenPAAndClient(PA, client) < problemDefinition.paMaxCoverageRadius:
            selected.append(client)
        else:
            unselected.append(client)

    return selected, unselected

def getDistanceBetweenPAAndClient(PA, client):
    xsDistance = PA.x - client.x
    ysDistance = PA.y - client.y
    return math.sqrt( xsDistance**2 + ysDistance**2 )

def getDistanceBetweenPAs(PA1, PA2):
    xsDistance = PA1.x - PA2.x
    ysDistance = PA1.y - PA2.y
    return math.sqrt( xsDistance**2 + ysDistance**2 )

def selectClientsUntilCapacity(pa, clients, problemDefinition):
    capacity = pa.capacity
    selected, unselected = [], []

    for client in clients:
        if capacity + client.band_width > problemDefinition.paMaxCapacity:
            unselected.append(client)
        else:
            selected.append(client)
            capacity += client.band_width

    return selected, unselected

# Para tentar minimizar a distância, vamos alocar no máximo (256/25 ~= 11) clientes por pa
def selectClientsUntilMaxAmount(pa, clients, problemDefinition):
    capacity = pa.capacity
    selected, unselected = [], []

    for client in clients:
        if (capacity + client.band_width > problemDefinition.paMaxCapacity) or len(selected) == 30:
            unselected.append(client)
        else:
            selected.append(client)
            capacity += client.band_width

    return selected, unselected

def minimizeTotalDistanceBetwenEnabledPAsAndClients(problemDefinition):

    def getSelectedClientsTotalDistance(PA, clients, problemDefinition):
            selected, unselected = [], []
            totalDistance = 0

            for client in clients:
                distance = getDistanceBetweenPAAndClient(PA, client)
                if distance < problemDefinition.paMaxCoverageRadius:
                    totalDistance += distance
                    selected.append(client)
                else:
                    unselected.append(client)

            return totalDistance

    def sortPAsForMinDistance(PAs, clients, problemDefinition):
        return sorted(PAs, key = lambda pa: getSelectedClientsTotalDistance(pa, clients, problemDefinition), reverse = True)

    finalPAsList = []
    candidatePAs = sortPAsForMinDistance(problemDefinition.PAs, problemDefinition.clients, problemDefinition)
    unnalocatedClients = [*problemDefinition.clients]
    while len(unnalocatedClients):
        # Pega o primeiro PA pq ele sempre é o que tem menor distância relativa
        currentPa = candidatePAs.pop(0)
        # Aloca o máximo de clientes possíveis no PA atual
        finalPAsList, unnalocatedClients = allocateClientesToPAs((finalPAsList, unnalocatedClients), currentPa, problemDefinition)
        # Reordena os PAs em função da distância relativa
        candidatePAs = sortPAsForMinDistance(candidatePAs, unnalocatedClients, problemDefinition)
    return finalPAsList + candidatePAs

def getTotalDistanceSumBetweenPAsAndClients(PAs):
    return sum(sum(getDistanceBetweenPAAndClient(pa, client) for client in pa.clients) for pa in PAs)

def shake(currentSolution, neighborhoodStrategyIndex, problemDefinition):
    if neighborhoodStrategyIndex == 1:
        return neighborhoodStrategyMoveClientToAnotherEnabledPA(currentSolution, problemDefinition)

    if neighborhoodStrategyIndex == 2:
        return neighborhoodStrategyExchangeClientBetweenPAs(currentSolution, problemDefinition)

    if neighborhoodStrategyIndex == 3:
        return neighborhoodStrategyToKillPaWithSmallerCapacityAndRedistributeClients(currentSolution, problemDefinition)

    if neighborhoodStrategyIndex == 4:
        return neighborhoodStrategyToKillRamdomPAAndRedistributeClients(currentSolution, problemDefinition)

    if neighborhoodStrategyIndex == 5:
        return neighborhoodStrategyRemoveClient(currentSolution, problemDefinition)

    return neighborhoodStrategyToKillPaWithSmallerCapacity(currentSolution, problemDefinition)

def neighborhoodStrategyToKillRamdomPAAndRedistributeClients(solution, problemDefinition):
    candidateSolution = copy.deepcopy(solution)
    currentPAs = candidateSolution.currentSolution

    pasLenght = len(currentPAs)
    if(pasLenght == 0):
        return solution

    paToKillIndex = random.randint(0,  pasLenght - 1)
    paToKill = currentPAs[paToKillIndex]
    currentPAs.remove(paToKill)

    sortedPAs = sortClosestPAs(currentPAs, paToKill)

    finalPAsList = []
    unnalocatedClients = paToKill.clients
    for pa in sortedPAs:
        pa.clients = []
        finalPAsList, unnalocatedClients = allocateClientesToPAs((finalPAsList, unnalocatedClients), pa, problemDefinition)
        if unnalocatedClients is None:
            break

    if percentageOfClientsNotAttendedBellowLimit(finalPAsList, problemDefinition):
        candidateSolution.currentSolution = finalPAsList
        return candidateSolution
    #  Se tiver algum cliente q n pode ser movido deixa ele sem atendimento mesmo pq tem q atender só 95%
    return solution

def neighborhoodStrategyToKillPaWithSmallerCapacityAndRedistributeClients(solution, problemDefinition):
    candidateSolution = copy.deepcopy(solution)
    currentPAs = candidateSolution.currentSolution

    pasSortedByCapacity = sorted(currentPAs, key = lambda pa: pa.capacity)
    if len(pasSortedByCapacity) == 0:
        return solution

    paToKill = pasSortedByCapacity[0]
    currentPAs.remove(paToKill)

    sortedPAs = sortClosestPAs(currentPAs, paToKill)

    finalPAsList = []
    unnalocatedClients = paToKill.clients
    for pa in sortedPAs:
        pa.clients = []
        finalPAsList, unnalocatedClients = allocateClientesToPAs((finalPAsList, unnalocatedClients), pa, problemDefinition)
        if unnalocatedClients is None:
            break

    if percentageOfClientsNotAttendedBellowLimit(finalPAsList, problemDefinition):
        candidateSolution.currentSolution = finalPAsList
        return candidateSolution
    #  Se tiver algum cliente q n pode ser movido deixa ele sem atendimento mesmo pq tem q atender só 95%
    return solution

def neighborhoodStrategyToKillPaWithSmallerCapacity(solution, problemDefinition):
    candidateSolution = copy.deepcopy(solution)
    currentPAs = candidateSolution.currentSolution

    pasSortedByCapacity = sorted(currentPAs, key = lambda pa: pa.capacity)
    if len(pasSortedByCapacity) == 0:
        return solution

    paToKill = pasSortedByCapacity[0]
    currentPAs.remove(paToKill)
    candidateSolution.currentSolution = currentPAs
    return candidateSolution

def sortClosestPAs(PAs, selectedPA):
    newPAs =  copy.deepcopy(PAs)
    sorted(newPAs, key = lambda pa: getDistanceBetweenPAs(selectedPA, pa))
    return newPAs

def neighborhoodStrategyMoveClientToAnotherEnabledPA(currentSolution, problemDefinition):
    candidateSolution = copy.deepcopy(currentSolution)
    currentPAs = filterPasWithClients(candidateSolution.currentSolution)

    pasLenght = len(currentPAs)
    if (pasLenght == 0):
        return candidateSolution

    originPAIndex = random.randint(0, pasLenght - 1)
    originPA = currentPAs[originPAIndex]

    destinyPAIndex = random.randint(0, pasLenght - 1)
    destinyPA = currentPAs[destinyPAIndex]

    clientsLenght = len(originPA.clients)
    if (clientsLenght == 0):
        return candidateSolution

    clientToMoveIndex = random.randint(0, clientsLenght - 1)
    clientToMove = originPA.clients[clientToMoveIndex]

    if destinyPA.capacity < problemDefinition.paMaxCapacity and getDistanceBetweenPAAndClient(destinyPA, clientToMove) <= problemDefinition.paMaxCoverageRadius:
        originPA.clients.remove(clientToMove)
        destinyPA.clients.append(clientToMove)

    currentPAs[originPAIndex] = originPA
    currentPAs[destinyPAIndex] = destinyPA
    candidateSolution.currentSolution = currentPAs
    return candidateSolution

def neighborhoodStrategyExchangeClientBetweenPAs(currentSolution, problemDefinition):
    candidateSolution = copy.deepcopy(currentSolution)
    currentPAs = filterPasWithClients(candidateSolution.currentSolution)

    pasLenght = len(currentPAs)
    if (pasLenght == 0):
        return candidateSolution
    indexPaA = random.randint(0, pasLenght - 1)
    paA = currentPAs[indexPaA]

    indexPaB = random.randint(0, pasLenght - 1)
    paB = currentPAs[indexPaB]

    clientALen = len(paA.clients)
    if (clientALen == 0):
        return candidateSolution
    indexClientA = random.randint(0, clientALen - 1)
    clientA = paA.clients[indexClientA]

    clientBLen = len(paB.clients)
    if (clientBLen == 0):
        return candidateSolution
    indexClientB = random.randint(0, clientBLen - 1)
    clientB = paB.clients[indexClientB]

    paBNewCapacity = paB.capacity - clientB.band_width + clientA.band_width
    paANewCapacity = paA.capacity - clientA.band_width + clientB.band_width

    distancePaBClientA = getDistanceBetweenPAAndClient(paB, clientA)
    distancePaAClientB = getDistanceBetweenPAAndClient(paA, clientB)

    if paBNewCapacity <= problemDefinition.paMaxCapacity and paANewCapacity <= problemDefinition.paMaxCapacity and distancePaAClientB <= problemDefinition.paMaxCoverageRadius and distancePaBClientA <= problemDefinition.paMaxCoverageRadius:
        paA.clients.remove(clientA)
        paA.clients.append(clientB)

        paB.clients.remove(clientB)
        paB.clients.append(clientA)

    currentPAs[indexPaA] = paA
    currentPAs[indexPaB] = paB
    candidateSolution.currentSolution = currentPAs
    return candidateSolution

def neighborhoodStrategyRemoveClient(currentSolution, problemDefinition):
    candidateSolution = copy.deepcopy(currentSolution)
    currentPAs = filterPasWithClients(candidateSolution.currentSolution)

    pasLenght = len(currentPAs)
    if (pasLenght == 0):
        return candidateSolution
    pasRange = random.randint(0, pasLenght - 1)

    indexPa = random.randint(0, pasRange)
    pa = currentPAs[indexPa]

    clientsLenght = len(pa.clients)
    indexClient = clientsLenght if clientsLenght == 0 else random.randint(0, clientsLenght - 1)
    client = pa.clients[indexClient]

    pa.clients.remove(client)

    currentPAs[indexPa] = pa
    candidateSolution.currentSolution = currentPAs
    return candidateSolution


def neighborhoodChange(solution, candidateSolution, neighborhoodStrategyIndex):

    # verifica se a solução y deve ser escolhida (Stochastic Ranking)
    def shouldSelectCandidateSolution(solution, candidateSolution, probability = 0.4):
        if random.random() <= probability:
            return True
        if not solution.feasible and candidateSolution.feasible :
            return True
        if solution.feasible and  not candidateSolution.feasible:
            return False
        return candidateSolution.violation < solution.violation


    if shouldSelectCandidateSolution(solution, candidateSolution):
        return candidateSolution, 1
    return solution, neighborhoodStrategyIndex + 1

def findMinPABestNeighbor(solution, problemDefinition, weights, objFunction):
    currentSolution = copy.deepcopy(solution)

    neighborhood1 = neighborhoodStrategyToKillPaWithSmallerCapacityAndRedistributeClients(currentSolution, problemDefinition)
    solution1 = objFunction(neighborhood1, problemDefinition, weights)
    cost1 = solution1.singleObjectiveValue + solution1.violation

    neighborhood2 = neighborhoodStrategyToKillRamdomPAAndRedistributeClients(currentSolution, problemDefinition)
    solution2 = objFunction(neighborhood2, problemDefinition, weights)
    cost2 = solution2.singleObjectiveValue + solution2.violation

    neighborhood3 = neighborhoodStrategyToKillPaWithSmallerCapacity(currentSolution, problemDefinition)
    solution3 = objFunction(neighborhood3, problemDefinition, weights)
    cost3 = solution3.singleObjectiveValue + solution3.violation

    neighborhoods = [neighborhood1, neighborhood2, neighborhood3]
    costs = [cost1, cost2, cost3]

    bestNeighbor = min(zip(neighborhoods, costs), key=lambda x: x[1])
    return bestNeighbor[0], bestNeighbor[1]


def findBestNeighbor(solution, problemDefinition, weights, objFunction):
    currentSolution = copy.deepcopy(solution)

    neighborhood1 = neighborhoodStrategyToKillPaWithSmallerCapacityAndRedistributeClients(currentSolution, problemDefinition)
    solution1 = objFunction(neighborhood1, problemDefinition, weights)
    cost1 = solution1.singleObjectiveValue + solution1.violation

    neighborhood2 = neighborhoodStrategyToKillRamdomPAAndRedistributeClients(currentSolution, problemDefinition)
    solution2 = objFunction(neighborhood2, problemDefinition, weights)
    cost2 = solution2.singleObjectiveValue + solution2.violation

    neighborhood3 = neighborhoodStrategyToKillPaWithSmallerCapacity(currentSolution, problemDefinition)
    solution3 = objFunction(neighborhood3, problemDefinition, weights)
    cost3 = solution3.singleObjectiveValue + solution3.violation

    neighborhood4 = neighborhoodStrategyMoveClientToAnotherEnabledPA(currentSolution, problemDefinition)
    solution4 = objFunction(neighborhood4, problemDefinition, weights)
    cost4 = solution4.singleObjectiveValue + solution4.violation

    neighborhood5 = neighborhoodStrategyExchangeClientBetweenPAs(currentSolution, problemDefinition)
    solution5 = objFunction(neighborhood5, problemDefinition, weights)
    cost5 = solution5.singleObjectiveValue + solution5.violation

    neighborhood6 = neighborhoodStrategyRemoveClient(currentSolution, problemDefinition)
    solution6 = objFunction(neighborhood6, problemDefinition, weights)
    cost6 = solution6.singleObjectiveValue + solution6.violation

    neighborhoods = [neighborhood1, neighborhood2, neighborhood3, neighborhood4, neighborhood5, neighborhood6]
    costs = [cost1, cost2, cost3, cost4, cost5, cost6]

    bestNeighbor = min(zip(neighborhoods, costs), key=lambda x: x[1])
    return bestNeighbor[0]

def firstImprovement(solution, problemDefinition, weights, objFunction):
    currentSolutionCost = solution.singleObjectiveValue + solution.violation
    bestNeighbor, bestNeighborCost = findMinPABestNeighbor(solution, problemDefinition, weights, objFunction)
    if bestNeighborCost < currentSolutionCost:
        return bestNeighbor
    return solution

def bvnsToMinimizeAmountOfClients(problemDefinition, weights, objFunction):
    numberOfEvaluatedCandidates = 0
    max_num_sol_avaliadas = 5
    kmax = 6

    weights = np.random.random(size = 2)
    normalizedWeights = weights/sum(weights)

    solution = factoryInitialSolutionToMinimizePAsAmount(problemDefinition)
    solution = objFunction(solution, problemDefinition, weights)

    numberOfEvaluatedCandidates += 1

    while numberOfEvaluatedCandidates < max_num_sol_avaliadas:

        k = 1
        while k <= kmax:

            # Gera uma solução candidata na k-ésima vizinhança de x
            y = shake(solution, k, problemDefinition)
            y = findBestNeighbor(solution, problemDefinition, weights, objFunction)
            y = objFunction(y, problemDefinition, weights)
            numberOfEvaluatedCandidates += 1

            # Atualiza solução corrente e estrutura de vizinhança (se necessário)
            solution, k = neighborhoodChange(solution, y, k)

    return solution

def nonDominatedSolutions(history):
    nonDominatedSolutionsF1  = []
    nonDominatedSolutionsF2  = []

    for i in range(len(history)):
        currentSolution = (history[i].fitness, history[i].secondObjectiveFitness)
        dominated = False
        for j in range(len(history)):
            if ((currentSolution[0] > history[j].fitness) and (currentSolution[1] >= history[j].secondObjectiveFitness)) or ((currentSolution[1] > history[j].secondObjectiveFitness) and (currentSolution[0] >= history[j].fitness)):
                dominated = True
                break

        if not dominated:
            nonDominatedSolutionsF1.append(currentSolution[0])
            nonDominatedSolutionsF2.append(currentSolution[1])

    return nonDominatedSolutionsF1, nonDominatedSolutionsF2


def mapSolutions(history):
    unfeasibleSolutionsF1 = []
    unfeasibleSolutionsF2 = []
    feasibleSolutionsF1 = []
    feasibleSolutionsF2 = []
    for solution in history:
        if solution.violation > 0:
            unfeasibleSolutionsF1.append(solution.fitness)
            unfeasibleSolutionsF2.append(solution.secondObjectiveFitness)
        else:
            feasibleSolutionsF1.append(solution.fitness)
            feasibleSolutionsF2.append(solution.secondObjectiveFitness)
    return unfeasibleSolutionsF1, unfeasibleSolutionsF2, feasibleSolutionsF1, feasibleSolutionsF2

def pwStrategy():
    problemDefinition = ProblemDefinition()
    problemDefinition.clients = getClients()
    problemDefinition.PAs = factoryPAs()

    amountOfParetoOptimalSolutions = 20

    history = []
    for i in range(amountOfParetoOptimalSolutions):
        weights = np.random.random(size = 2)
        normalizedWeights = weights/sum(weights)

        solution = bvnsToMinimizeAmountOfClients(problemDefinition, weights, pw)
        history.append(solution)

    nonDominatedSolutionsF1, nonDominatedSolutionsF2 = nonDominatedSolutions(history)
    unfeasibleSolutionsF1, unfeasibleSolutionsF2, feasibleSolutionsF1, feasibleSolutionsF2 = mapSolutions(history)

    plt.plot(nonDominatedSolutionsF1, nonDominatedSolutionsF2,'ks',markerfacecolor='none',markersize=10)
    plt.plot(unfeasibleSolutionsF1, unfeasibleSolutionsF2,'b.')
    plt.plot(feasibleSolutionsF1, feasibleSolutionsF2, 'r.')
    plt.legend(['Fronteira Pareto Estimada','Soluções Inviáveis', 'Soluções Factíveis'])
    plt.title('Soluções estimadas - pw')
    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.show()

def peStrategy():
    problemDefinition = ProblemDefinition()
    problemDefinition.clients = getClients()
    problemDefinition.PAs = factoryPAs()

    amountOfParetoOptimalSolutions = 20

    history = []
    for i in range(amountOfParetoOptimalSolutions):
        solution = bvnsToMinimizeAmountOfClients(problemDefinition, [1, 1], pe)
        history.append(solution)

    nonDominatedSolutionsF1, nonDominatedSolutionsF2 = nonDominatedSolutions(history)
    unfeasibleSolutionsF1, unfeasibleSolutionsF2, feasibleSolutionsF1, feasibleSolutionsF2 = mapSolutions(history)

    plt.plot(nonDominatedSolutionsF1, nonDominatedSolutionsF2,'ks',markerfacecolor='none',markersize=10)
    plt.plot(unfeasibleSolutionsF1, unfeasibleSolutionsF2,'b.')
    plt.plot(feasibleSolutionsF1, feasibleSolutionsF2, 'r.')
    plt.legend(['Fronteira Pareto Estimada','Soluções Inviáveis', 'Soluções Factíveis'])
    plt.title('Soluções estimadas - pe')
    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.show()

def multiObjectiveMain():
    #for i in range(5):
    pwStrategy()
    #for i in range(5):
    peStrategy()

multiObjectiveMain()