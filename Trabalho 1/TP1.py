import rvns
import csv
import math
from functools import reduce

CLIENTS_MIN_COVERAGE = 0.95

class Client:
    def __init__(self, x, y, band_width):
        self.x = x
        self.y = y
        self.band_width = band_width

PA_MAX_CAPACITY = 54
PA_MAX_COVERAGE_RADIUS = 70
MAX_AMOUNT_OF_PAS = 25

# Não sei de onde vem essa informação
NUMBER_OF_PLACES_TO_INSTALL_PAS = 400 / 5

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


def factoryPAs():
    PAs = []
    for x in range(0, 400, 5):
        for y in range(0, 400, 5):
            newPA = PA(x, y)
            PAs.append(newPA)
    return PAs


def minimizePAsHeuristic(PAs, clients):
    def selectClientsUntilCapacity(clients):
        capacity = 0
        selected, unselected = [], []

        for client in clients:
            if capacity + client.band_width > PA_MAX_CAPACITY:
                unselected.append(client)
            else:
                selected.append(client)
                capacity += client.band_width

        return selected, unselected

    def selectClientsByDistance(PA, clients):
        selected, unselected = [], []
        
        clientsAndTheirDistance = list(map(lambda client: (client,  getDistanceBetweenPAAndClient(PA, client)), clients))
        sortedClientsAndTheirDistance  = sorted(clientsAndTheirDistance, key = lambda clientAndItsDistance: clientAndItsDistance[1], reverse = True)

        for sortedClientsAndItsDistance in sortedClientsAndTheirDistance:
            if sortedClientsAndItsDistance[1] < PA_MAX_COVERAGE_RADIUS:
                selected.append(sortedClientsAndItsDistance[0])
            else:
                unselected.append(sortedClientsAndItsDistance[0])

        return selected, unselected


    def allocateClientesToPAs(PAsAndClients, currentPA):
        PAs, unnalocatedClients = PAsAndClients

        candidateClients, tooFarClients = selectClientsByDistance(currentPA, unnalocatedClients)
        selectedClients, restClients = selectClientsUntilCapacity(candidateClients)
        newPA = currentPA.withNewClients(selectedClients)
        return ([*PAs, newPA], restClients + tooFarClients)

    return reduce(allocateClientesToPAs, PAs, ([], clients))

def printPAs(PAs):
    for PA in PAs:
        print(f"PA capacity: {PA.capacity}")
        print(f"PA amount of clients: {len(PA.clients)}")

def getClients(): 
    clients = []
    with open('clientes.csv', mode='r') as clientes_csv:

        leitor_csv = csv.reader(clientes_csv)
        
        for linha in leitor_csv:
            newClient = Client(float(linha[0]), float(linha[1]), float(linha[2]))
            clients.append(newClient)
    return clients


def readClients(clients):
    for client in clients:
        print(f"{client.x}\n")

def getDistanceBetweenPAAndClient(PA, client):
    xsDistance = PA.x - client.x
    ysDistance = PA.y - client.y
    return math.sqrt( xsDistance**2 + ysDistance**2 )

def filterPAsEnabled(PAs):
    return list(filter(lambda PA: PA.capacity != 0, PAs))


def minimizeTotalDistanceSum(PAs, clients):
    totalDistance = 0
    for PA in PAs:
        for client in clients:
            distance = getDistanceBetweenPAAndClient(PA, client)

def main():
    clients = getClients()

    numberOfClients = len(clients)
    print(f"Number of clients: {numberOfClients}\n")
    
    PAs = factoryPAs()
    updatedPAs = minimizePAsHeuristic(PAs, clients)

    PAsEnabled = filterPAsEnabled(updatedPAs[0])
    print(f"Number of PAs: {len(PAsEnabled)}\n")


main()