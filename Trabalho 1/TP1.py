import rvns
import csv
import math
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
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
        currentSolution = None 
    ):
        self.fitness = fitness
        self.violation = violation
        self.feasible = feasible
        self.currentSolution = [] if currentSolution is None else currentSolution

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

def filterPAsEnabled(PAs):
    return list(filter(lambda PA: PA.capacity != 0, PAs))

def factoryInitialSolutionToMinimizePAsAmount(problemDefinition):
    solution = Solution()
    initialPAsList = minimizePAsHeuristic(problemDefinition)
    solution.currentSolution = filterPAsEnabled(initialPAsList)
    return solution

def calculateViolation(solution, problemDefinition):
    R1 = percentageOfClientsNotAttended(solution.currentSolution, problemDefinition)
    print("percentageOfClientsNotAttended", R1)
    R2 = amountOfPAsAboveCapacity(solution.currentSolution, problemDefinition)
    print("amountOfPAsAboveCapacity", R2)
    R3 = amountOfPAsClientOutsideMaxRange(solution.currentSolution, problemDefinition)
    print("amountOfPAsClientOutsideMaxRange", R3)
    R4 = amountOfReplicatedClients(solution.currentSolution)
    print("amountOfReplicatedClients", R4)
    R5 = amountOfPAsAboveMaxAmount(solution.currentSolution, problemDefinition)
    print("amountOfPAsAboveMaxAmount", R5)
    return 150 * (R1 + R2 + R3 + R4 + R5)

def objectiveFuntionMinimizeAmountOfPAs(solution, problemDefinition):
    newSolution = copy.deepcopy(solution)
    enabledPAs = filterPAsEnabled(newSolution.currentSolution)

    newSolution.fitness = len(enabledPAs)
    
    newSolution.violation = calculateViolation(solution, problemDefinition)
    
    newSolution.feasible = solution.violation == 0

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
def percentageOfClientsNotAttended(PAs, problemDefinition):
    selectedClients = set()

    for pa in PAs:
        selectedClients.update(pa.clients)
    
    perentageOfClientsAttended = len(selectedClients)/len(problemDefinition.clients)
    return (1 - perentageOfClientsAttended )* 100

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

def shakeToMinimizeDistance(currentSolution, neighborhoodStrategyIndex, problemDefinition):
    
    # Passar um único cliente para outro PA aleatório ativo
    if neighborhoodStrategyIndex == 1:
       return neighborhoodStrategyMoveClientToAnotherEnabledPA(currentSolution, problemDefinition)
    # Passar dois clientes para outro PA aleatório
    if neighborhoodStrategyIndex == 2:
        newSolution = neighborhoodStrategyMoveClientToAnotherEnabledPA(currentSolution, problemDefinition)
        return neighborhoodStrategyMoveClientToAnotherEnabledPA(newSolution, problemDefinition)
    # Trocar clientes de PAs
    return neighborhoodStrategyExchangeClientBetweenPAs(currentSolution, problemDefinition)

def shakeToMinimizeAmountOfPAs(currentSolution, neighborhoodStrategyIndex, problemDefinition):
    
    # Mata um pa e realoca os clientes para os demais
    if neighborhoodStrategyIndex == 1:
        return neighborhoodStrategyToKillPaWithSmallerCapacity(currentSolution, problemDefinition)
    # Passar dois clientes para outro PA aleatório
    if neighborhoodStrategyIndex == 2:
        return neighborhoodStrategyToKillRamdomPAAndRedistributeClients(currentSolution, problemDefinition)
    # Trocar clientes de PAs
    return neighborhoodStrategyToKillRamdomPA(currentSolution, problemDefinition)

def neighborhoodStrategyToKillRamdomPAAndRedistributeClients(solution, problemDefinition):
    candidateSolution = copy.deepcopy(solution)
    currentPAs = candidateSolution.currentSolution

    paToKillIndex = random.randint(0, len(currentPAs) - 1)
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
    
    if percentageOfClientsNotAttended(finalPAsList, problemDefinition): 
        candidateSolution.currentSolution = finalPAsList
        return candidateSolution
    #  Se tiver algum cliente q n pode ser movido deixa ele sem atendimento mesmo pq tem q atender só 95%
    return solution

def neighborhoodStrategyToKillPaWithSmallerCapacityAndRedistributeClients(solution, problemDefinition):
    candidateSolution = copy.deepcopy(solution)
    currentPAs = candidateSolution.currentSolution

    pasSortedByCapacity = sorted(currentPAs, key = lambda pa: pa.capacity)
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
    
    if percentageOfClientsNotAttended(finalPAsList, problemDefinition): 
        candidateSolution.currentSolution = finalPAsList
        return candidateSolution
    #  Se tiver algum cliente q n pode ser movido deixa ele sem atendimento mesmo pq tem q atender só 95%
    return solution

def sortClosestPAs(PAs, selectedPA):
    newPAs =  copy.deepcopy(PAs)
    sorted(newPAs, key = lambda pa: getDistanceBetweenPAs(selectedPA, pa))
    return newPAs

def neighborhoodStrategyMoveClientToAnotherEnabledPA(currentSolution, problemDefinition):
    candidateSolution = copy.deepcopy(currentSolution)
    currentPAs = candidateSolution.currentSolution

    originPAIndex = random.randint(0, len(currentPAs) - 1)
    originPA = currentPAs[originPAIndex]

    destinyPAIndex = random.randint(0, len(currentPAs) - 1)
    destinyPA = currentPAs[destinyPAIndex]

    clientToMoveIndex = random.randint(0, len(originPA.clients) - 1)
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
    currentPAs = candidateSolution.currentSolution

    indexPaA = random.randint(0, len(currentPAs) - 1)
    paA = currentPAs[indexPaA]

    indexPaB = random.randint(0, len(currentPAs) - 1)
    paB = currentPAs[indexPaB]

    indexClientA = random.randint(0, len(paA.clients) - 1)
    clientA = paA.clients[indexClientA]

    indexClientB = random.randint(0, len(paB.clients) - 1)
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

def neighborhoodChange(solution, candidateSolution, neighborhoodStrategyIndex):

    # verifica se a solução y deve ser escolhida (Stochastic Ranking)
    def shouldSelectCandidateSolution(solution, candidateSolution, probability = 0.4):
        if((solution.feasible and candidateSolution.feasible) or random.random() <= probability):
            return candidateSolution.fitness < solution.fitness
        
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

def bvnsToMinimizeAmountOfClients(problemDefinition):
    numberOfEvaluatedCandidates = 0
    max_num_sol_avaliadas = 5
    kmax = 3

    solution = factoryInitialSolutionToMinimizePAsAmount(problemDefinition)
    solution = objectiveFuntionMinimizeAmountOfPAs(solution, problemDefinition)

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
            y = shakeToMinimizeAmountOfPAs(solution, k, problemDefinition)     
            y = objectiveFuntionMinimizeAmountOfPAs(y, problemDefinition)
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
                    
    plotBestSolution(bestuptonow)
    plotFirstSolution(bestuptonow)
    plotSolution(historico, bestuptonow)
    
def bvnsToMinimizeTotalDistance(problemDefinition):
    numberOfEvaluatedCandidates = 0
    max_num_sol_avaliadas = 5
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
                    
    plotBestSolution(bestuptonow)
    plotFirstSolution(bestuptonow)
    plotSolution(historico, bestuptonow)

def main():
    problemDefinition = ProblemDefinition()
    problemDefinition.clients = getClients()
    problemDefinition.PAs = factoryPAs()

    bvnsToMinimizeAmountOfClients(problemDefinition)
    bvnsToMinimizeTotalDistance(problemDefinition)

main()
