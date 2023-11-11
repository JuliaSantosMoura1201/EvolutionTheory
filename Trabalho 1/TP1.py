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

def luana():
    problemDefinition = ProblemDefinition()
    problemDefinition.clients = getClients()
    problemDefinition.PAs = factoryPAs()

    pas_x = [pa.x for pa in problemDefinition.PAs] 
    pas_y = [pa.y for pa in problemDefinition.PAs]
    plt.scatter(pas_x, pas_y, label='Possíveis Pas', color='red', marker='o')

    newPas = []
    for i in range(0, len(problemDefinition.PAs), 16):
        newPas.append(problemDefinition.PAs[i])

    finalPas = []
    for i in range(0, len(newPas), 16):
        finalPas.append(newPas[i])

    newPas_x = [pa.x for pa in finalPas] 
    newPas_y = [pa.y for pa in finalPas]
    plt.scatter(newPas_x, newPas_y, label='Pas selecionados', color='green', marker='o')

    clients_x = [client.x for client in problemDefinition.clients] 
    clients_y = [client.y for client in problemDefinition.clients]
    #plt.scatter(clients_x, clients_y, label='Clientes', color='blue', marker='o')

    plt.xlim(0, 400)
    plt.ylim(0, 400)

    plt.xlabel('Eixo X')
    plt.ylabel('Eixo Y')

    plt.show()

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
    initialPAsList = minimizePAsHeuristic(problemDefinition)
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
    newSolution.violation = newSolution.violation + 150 * violation
    newSolution.singleObjectiveValue = newSolution.fitness + newSolution.violation
    return newSolution

def factoryInitialSolutionToMinimizeTotalDistance(problemDefinition):
    solution = Solution()
    initialPAsList = minimizeTotalDistanceBetwenEnabledPAsAndClients(problemDefinition)
    solution.currentSolution = filterPAsEnabled(initialPAsList)
    return solution

def objectiveFuntionToMinimizeTotalDistance(solution, problemDefinition):
    newSolution = copy.deepcopy(solution)
    enabledPAs = filterPAsEnabled(newSolution.currentSolution)

    newSolution.fitness = getTotalDistanceSumBetweenPAsAndClients(enabledPAs)
    
    newSolution.violation = calculateViolation(solution, problemDefinition)
    
    newSolution.feasible = solution.violation == 0

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

def minimizePAsHeuristic(problemDefinition):
    
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

def printPAs(PAs):
    for PA in PAs:
        print(f"PA capacity: {PA.capacity}")
        print(f"PA amount of clients: {len(PA.clients)}")

def readClients(clients):
    for client in clients:
        print(f"{client.x}\n")

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

def shakeToMinimizeDistance(currentSolution, neighborhoodStrategyIndex, problemDefinition):
    
    if neighborhoodStrategyIndex == 1:
       return neighborhoodStrategyMoveClientToAnotherEnabledPA(currentSolution, problemDefinition)
    
    if neighborhoodStrategyIndex == 2:
        return neighborhoodStrategyExchangeClientBetweenPAs(currentSolution, problemDefinition)
   
    return neighborhoodStrategyRemoveClient(currentSolution, problemDefinition)

def shakeToMinimizeAmountOfPAs(currentSolution, neighborhoodStrategyIndex, problemDefinition):

    if neighborhoodStrategyIndex == 1:
        return neighborhoodStrategyToKillPaWithSmallerCapacityAndRedistributeClients(currentSolution, problemDefinition)
    
    if neighborhoodStrategyIndex == 2:
        return neighborhoodStrategyToKillRamdomPAAndRedistributeClients(currentSolution, problemDefinition)
    
    return neighborhoodStrategyToKillPaWithSmallerCapacity(currentSolution, problemDefinition)

def neighborhoodStrategyToKillRamdomPAAndRedistributeClients(solution, problemDefinition):
    candidateSolution = copy.deepcopy(solution)
    currentPAs = candidateSolution.currentSolution

    pasLenght = len(currentPAs)
    paToKillIndex = pasLenght if pasLenght == 0 else random.randint(0,  pasLenght - 1)
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
    
    return candidateSolution

def neighborhoodStrategyToKillPaWithSmallerCapacityAndRedistributeClients(solution, problemDefinition):
    candidateSolution = copy.deepcopy(solution)
    currentPAs = candidateSolution.currentSolution

    pasSortedByCapacity = sorted(currentPAs, key = lambda pa: pa.capacity)
    if pasSortedByCapacity is None:
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
    
    return candidateSolution

def neighborhoodStrategyToKillPaWithSmallerCapacity(solution, problemDefinition):
    candidateSolution = copy.deepcopy(solution)
    currentPAs = candidateSolution.currentSolution

    pasSortedByCapacity = sorted(currentPAs, key = lambda pa: pa.capacity)
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
    originPAIndex = pasLenght if pasLenght == 0 else random.randint(0, pasLenght - 1)
    originPA = currentPAs[originPAIndex]

    destinyPAIndex = pasLenght if pasLenght == 0 else random.randint(0, pasLenght - 1)
    destinyPA = currentPAs[destinyPAIndex]

    clientsLenght = len(originPA.clients)
    clientToMoveIndex = clientsLenght if clientsLenght == 0 else random.randint(0, clientsLenght - 1)
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
    indexPaA = 0 if pasLenght == 0 else random.randint(0, pasLenght - 1)
    paA = currentPAs[indexPaA]

    indexPaB = 0 if pasLenght == 0 else random.randint(0, pasLenght - 1)
    paB = currentPAs[indexPaB]

    clientALen = len(paA.clients)
    indexClientA = 0 if clientALen == 0 else random.randint(0, clientALen - 1)
    clientA = paA.clients[indexClientA]

    clientBLen = len(paB.clients)
    indexClientB = 0 if clientBLen == 0 else random.randint(0, clientBLen - 1)
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
    pasRange =  pasLenght if pasLenght == 0 else random.randint(0, pasLenght - 1)

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

def plotSolution(historico, bestuptonow):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    s = len(historico.fit)
    ax1.plot(np.linspace(0,s-1,s),historico.fit,'k-')
    ax2.plot(np.linspace(0,s-1,s),bestuptonow.fit,'b:')
    fig.suptitle('Evolução da qualidade da solução candidata')
    ax1.set_ylabel('current fitness(x)')
    ax2.set_ylabel('best fitness(x)')
    ax2.set_xlabel('Número de avaliações')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
    plt.show()

def plotBestSolution(bestuptonow):
    print('\n--- MELHOR SOLUÇÃO ENCONTRADA ---\n')
    print('Identificação dos projetos selecionados:\n')
    print('x = {}\n'.format(bestuptonow.sol[-1]))
    print('fitness(x) = {:.2f}\n'.format(bestuptonow.fit[-1]))
    print('violation(x) = {:.2f}\n'.format(bestuptonow.vio[-1]))
    print('feasible(x) = {}\n'.format(bestuptonow.fea[-1]))

def plotFirstSolution(bestuptonow):
    print('\n--- SOLUÇÃO INICIAL CONSTRUÍDA ---\n')
    print('Identificação dos projetos selecionados:\n')
    print('Mensagem', bestuptonow.vio[0])
    print('x = {}\n'.format(bestuptonow.sol[0]))
    print('fitness(x) = {:.2f}\n'.format(bestuptonow.fit[0]))
    print('violation(x) = {:.2f}\n'.format(bestuptonow.vio[0]))
    print('feasible(x) = {}\n'.format(bestuptonow.fea[0]))

def findMinDistBestNeighbor(solution, problemDefinition):
    currentSolution = copy.deepcopy(solution)

    neighborhood1 = neighborhoodStrategyMoveClientToAnotherEnabledPA(currentSolution, problemDefinition)
    solution1 = objectiveFuntionToMinimizeTotalDistance(neighborhood1, problemDefinition)
    cost1 = solution1.singleObjectiveValue + solution1.violation

    neighborhood2 = neighborhoodStrategyExchangeClientBetweenPAs(currentSolution, problemDefinition)
    solution2 = objectiveFuntionToMinimizeTotalDistance(neighborhood2, problemDefinition)
    cost2 = solution2.singleObjectiveValue + solution2.violation
    
    neighborhood3 = neighborhoodStrategyRemoveClient(currentSolution, problemDefinition)
    solution3 = objectiveFuntionToMinimizeTotalDistance(neighborhood3, problemDefinition)
    cost3 = solution3.singleObjectiveValue + solution3.violation

    neighborhoods = [neighborhood1, neighborhood2, neighborhood3]
    costs = [cost1, cost2, cost3]

    bestNeighbor = min(zip(neighborhoods, costs), key=lambda x: x[1])
    return bestNeighbor[0], bestNeighbor[1]

def firstImprovementMinimizeDist(solution, problemDefinition):
    currentSolutionCost = solution.fitness + solution.violation
    bestNeighbor, bestNeighborCost = findMinDistBestNeighbor(solution, problemDefinition)
    if bestNeighborCost < currentSolutionCost:
        return bestNeighbor
    return solution 

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

def firstImprovementMinimizePas(solution, problemDefinition, weights, objFunction):
    currentSolutionCost = solution.singleObjectiveValue + solution.violation
    bestNeighbor, bestNeighborCost = findMinPABestNeighbor(solution, problemDefinition, weights, objFunction)
    if bestNeighborCost < currentSolutionCost:
        return bestNeighbor
    return solution 

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
    kmax = 3

    solution = factoryInitialSolutionToMinimizePAsAmount(problemDefinition)
    solution = objectiveFuntionMinimizeAmountOfPAs(solution, problemDefinition)

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
    
def bvnsToMinimizeTotalDistance(problemDefinition):
    numberOfEvaluatedCandidates = 0
    max_num_sol_avaliadas = 5000
    kmax = 3

    solution = factoryInitialSolutionToMinimizeTotalDistance(problemDefinition)
    solution = objectiveFuntionToMinimizeTotalDistance(solution, problemDefinition)

    numberOfEvaluatedCandidates += 1

    # Armazena dados para plot
    historico = History()
    historico.fit.append(solution.fitness)
    historico.sol.append(solution.currentSolution)
    historico.fea.append(solution.feasible)
    historico.vio.append(solution.violation)

    bestuptonow = History()
    bestuptonow.fit.append(solution.fitness)
    bestuptonow.sol.append(solution.currentSolution)
    bestuptonow.fea.append(solution.feasible)
    bestuptonow.vio.append(solution.violation)

    while numberOfEvaluatedCandidates < max_num_sol_avaliadas:

        k = 1
        while k <= kmax:

            # Gera uma solução candidata na k-ésima vizinhança de x 
            y = shakeToMinimizeDistance(solution, k, problemDefinition)     
            y = firstImprovementMinimizeDist(solution, problemDefinition) 
            y = objectiveFuntionToMinimizeTotalDistance(y, problemDefinition)
            numberOfEvaluatedCandidates += 1

            # Atualiza solução corrente e estrutura de vizinhança (se necessário)
            solution, k = neighborhoodChange(solution, y, k)

            # Armazena dados para plot
            historico.fit.append(solution.fitness)
            historico.sol.append(solution.currentSolution)
            historico.fea.append(solution.feasible)  
            historico.vio.append(solution.violation)

            # Mantém registro da melhor solução encontrada até então
            condition0 = solution.feasible == True and bestuptonow.fea[-1] == False
            condition1 = solution.feasible == True and bestuptonow.fea[-1] == True and solution.fitness < bestuptonow.fit[-1]
            condition2 = solution.feasible == False and bestuptonow.fea[-1] == False and solution.violation < bestuptonow.vio[-1]
            if condition0 or condition1 or condition2 == True:
                bestuptonow.fit.append(solution.fitness)
                bestuptonow.sol.append(solution.currentSolution)
                bestuptonow.fea.append(solution.feasible)
                bestuptonow.vio.append(solution.violation)
            else:
                bestuptonow.fit.append(bestuptonow.fit[-1])
                bestuptonow.sol.append(bestuptonow.sol[-1])
                bestuptonow.fea.append(bestuptonow.fea[-1])
                bestuptonow.vio.append(bestuptonow.vio[-1])
                    
    #plotBestSolution(bestuptonow)
    #plotFirstSolution(bestuptonow)
    #plotSolution(historico, bestuptonow)
    return bestuptonow

def plotBestSolutions(solutions):
    fig, ax1 = plt.subplots(1, 1)
    
    for solutuion in solutions:
        s = len(solutuion.fit)
        ax1.plot(np.linspace(0, s - 1, s), solutuion.fit, '-')
    
    fig.suptitle('Evolução do fitness')
    ax1.set_ylabel('Evolução do valor de f(·)')
    ax1.set_xlabel('Número de avaliações')
    
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()

def plotTheBestSolution(solution):
    import matplotlib.pyplot as plt

def plotTheBestSolution(solution):
    pas = solution.sol[-1]
    print(pas)
    # Defina uma lista de cores e símbolos para PAs e clientes
    colors_pa = [
        'red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown', 'cyan',
        'magenta', 'yellow', 'lime', 'indigo', 'teal', 'maroon', 'olive', 'navy',
        'salmon', 'sienna', 'turquoise', 'gold', 'violet', 'plum', 'slategray', 'darkgreen'
    ]
    current_color_index = 0  # Variável para alternar as cores
    
    for pa in pas:
        current_color = colors_pa[current_color_index]
        
        plt.scatter(pa.x, pa.y, s=50, color=current_color, marker='s', label='PA')
        for client in pa.clients:
            plt.scatter(client.x, client.y, s=20, color=current_color, marker='o', label='Client')

        current_color_index = (current_color_index + 1) % len(colors_pa)
    
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.show()


def printResults(historyList):
    fitValues = [history.fit for history in historyList]
    fitValues = [fit for sublist in fitValues for fit in sublist]

    fitMin = min(fitValues)
    fitMax = max(fitValues)
    fitStd = np.std(fitValues)

    print("Menor valor de aptidão:", fitMin)
    print("Maior valor de aptidão:", fitMax)
    print("Desvio padrão dos valores de aptidão:", fitStd)

def main():
    problemDefinition = ProblemDefinition()
    problemDefinition.clients = getClients()
    problemDefinition.PAs = factoryPAs()

    bestSolutionsMinPA = []
    for i in range(5):
        bestUpToNow = bvnsToMinimizeAmountOfClients(problemDefinition)
        plotTheBestSolution(bestUpToNow)
        bestSolutionsMinPA.append(bestUpToNow)
    
    printResults(bestSolutionsMinPA)
    plotBestSolutions(bestSolutionsMinPA)
    
    bestSolutionMinDist = []
    for i in range(5):
        bestUpToNow = bvnsToMinimizeTotalDistance(problemDefinition)
        plotTheBestSolution(bestUpToNow)
        bestSolutionMinDist.append(bestUpToNow)

    printResults(bestSolutionMinDist)
    plotBestSolutions(bestSolutionMinDist)

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
        print(solution.violation)
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

    solution = factoryInitialSolutionToMinimizePAsAmount(problemDefinition)
    solution = objectiveFuntionMinimizeAmountOfPAs(solution, problemDefinition)

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
        peStrategy()
    #for i in range(5):
        pwStrategy()

multiObjectiveMain()




