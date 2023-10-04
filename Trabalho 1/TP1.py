import rvns
import csv
import math
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import copy

class Struct:
    pass

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
        feasible = False,
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

def factoryInitialSolution(problemDefinition):
    solution = Solution()
    initialPAsList = minimizePAsHeuristic(problemDefinition)
    solution.currentSolution = filterPAsEnabled(initialPAsList)
    print("Initial Solution fitness", len(solution.currentSolution))
    return solution

def objectiveFuntionMinimizeAmountOfPAs(solution, problemDefinition):
    newSolution = copy.deepcopy(solution)
    enabledPAs = filterPAsEnabled(solution.currentSolution)

    newSolution.fitness = len(enabledPAs)

    R1 = percentageOfClientsAttended(solution.currentSolution, problemDefinition)
    R2 = amountOfPAsAboveCapacity(solution.currentSolution, problemDefinition)
    R3 = amountOfPAsClientOutsideMaxRange(solution.currentSolution, problemDefinition)
    R4 = amountOfReplicatedClients(solution.currentSolution)
    R5 = amountOfPAsAboveMaxAmount(solution.currentSolution, problemDefinition)
    
    newSolution.violation = 150 * (R1 + R2 + R3 + R4 + R5)
    
    newSolution.feasible = solution.violation != 0

    return newSolution

#R1
def percentageOfClientsAttended(PAs, problemDefinition):
    selectedClients = set()

    for pa in PAs:
        selectedClients.update(pa.clients)
    
    perentageOfClientsAttended = len(selectedClients)/len(problemDefinition.clients)
    return perentageOfClientsAttended * 100

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
            if getDistanceBetweenPAAndClient(pa, client) <= problemDefinition.paMaxCoverageRadius:
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
            amountOfReplicatedClients += count 
    
    return amountOfReplicatedClients

#R5
def amountOfPAsAboveMaxAmount(PAs, problemDefinition):
    return len(PAs)  - problemDefinition.maxAmountOfPAs

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
    selectedClients, restClients = selectClientsUntilCapacity(candidateClients, problemDefinition)

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

def selectClientsUntilCapacity(clients, problemDefinition):
        capacity = 0
        selected, unselected = [], []

        for client in clients:
            if capacity + client.band_width > problemDefinition.paMaxCapacity:
                unselected.append(client)
            else:
                selected.append(client)
                capacity += client.band_width

        return selected, unselected

def minimizeTotalDistanceBetwenEnabledPAsAndClients(PAs, clients):
    
    def getSelectedClientsTotalDistance(PA, clients):
            selected, unselected = [], []
            totalDistance = 0

            for client in clients:
                distance = getDistanceBetweenPAAndClient(PA, client)
                if distance < PA_MAX_COVERAGE_RADIUS:
                    totalDistance += distance
                    selected.append(client)
                else:
                    unselected.append(client)

            return totalDistance
    
    def sortPAsForMinDistance(PAs, clients):
        return sorted(PAs, key = lambda pa: getSelectedClientsTotalDistance(pa, clients), reverse = True)
    
    return reduce(allocateClientesToPAs, sortPAsForMinDistance(PAs, clients), ([], clients))

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
    # do something
    return currentSolution

def neighborhoodChange(solution, cadidateSolution, neighborhoodStrategyIndex):
    # do something
    return solution, neighborhoodStrategyIndex

def main():

    problemDefinition = ProblemDefinition()
    problemDefinition.clients = getClients()
    problemDefinition.PAs = factoryPAs()

    # Contador do número de soluções candidatas avaliadas
    numberOfEvaluatedCandidates = 0

    # Máximo número de soluções candidatas avaliadas
    max_num_sol_avaliadas = 5

    # Número de estruturas de vizinhanças definidas
    kmax = 3

    # Gera solução inicial
    solution = factoryInitialSolution(problemDefinition)

    # Avalia solução inicial
    solution = objectiveFuntionMinimizeAmountOfPAs(solution, problemDefinition)
    numberOfEvaluatedCandidates += 1

    # Armazena dados para plot
    historico = Struct()
    historico.fit = []
    historico.sol = []
    historico.fea = []
    historico.vio = []
    historico.fit.append(solution.fitness)
    historico.sol.append(solution.currentSolution)
    historico.fea.append(solution.feasible)
    historico.vio.append(solution.violation)

    bestuptonow = Struct()
    bestuptonow.fit = []
    bestuptonow.sol = []
    bestuptonow.fea = []
    bestuptonow.vio = []
    bestuptonow.fit.append(solution.fitness)
    bestuptonow.sol.append(solution.currentSolution)
    bestuptonow.fea.append(solution.feasible)
    bestuptonow.vio.append(solution.violation)

    # Ciclo iterativo do método
    while numberOfEvaluatedCandidates < max_num_sol_avaliadas:

        k = 1
        while k <= kmax:

            # Gera uma solução candidata na k-ésima vizinhança de x          
            y = shake(solution, k, problemDefinition)
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
                    
    print('\n--- SOLUÇÃO INICIAL CONSTRUÍDA ---\n')
    print('Identificação dos projetos selecionados:\n')
    print('Mensagem', bestuptonow.vio[0])
    print('x = {}\n'.format(bestuptonow.sol[0]))
    print('fitness(x) = {:.2f}\n'.format(bestuptonow.fit[0]))
    print('violation(x) = {:.2f}\n'.format(bestuptonow.vio[0]))
    print('feasible(x) = {}\n'.format(bestuptonow.fea[0]))

    print('\n--- MELHOR SOLUÇÃO ENCONTRADA ---\n')
    print('Identificação dos projetos selecionados:\n')
    print('x = {}\n'.format(bestuptonow.sol[-1]))
    print('fitness(x) = {:.2f}\n'.format(bestuptonow.fit[-1]))
    print('violation(x) = {:.2f}\n'.format(bestuptonow.vio[-1]))
    print('feasible(x) = {}\n'.format(bestuptonow.fea[-1]))

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

main()