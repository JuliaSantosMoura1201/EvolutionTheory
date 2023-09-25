import rvns
import csv
import math

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
        return sorted(clients, key = lambda client: getDistanceBetweenPAAndClient(self, client))[-1]

    def withNewClients(self, *clients):
        if not clients:
            return self
        newClients = self.clients + list(clients)
        return PA(
            x = self.x,
            y = self.y,
            coverage_radius = self._getClientWithBiggestDistance(newClients),
            capacity = self.capacity + sum([client.band_width for client in clients]),
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

def initialConstructiveHeuristic(PAs, clients):
    # ordenar os clientes pelo band width antes
    clientsIndex = 0
    updatedPAs = []
    for paIndex in range(len(PAs)):
        newPA = PAs[paIndex]
        while clientsIndex < len(clients):
            if(newPA.capacity + clients[clientsIndex].band_width >= PA_MAX_CAPACITY):
                break
            newPA = newPA.withNewClients(clients[clientsIndex])
            clientsIndex += 1
        print(f"PA capacity: {newPA.capacity}")
        print(f"PA clients amount: {len(newPA.clients)}")
        updatedPAs.append(newPA)
    return updatedPAs

def printPAs(PAs):
    for PA in PAs:
        print(f"PA capacity: {PA.capacity}", end = '')
        print(f"PA amount of clients: {len(PA.clients)}", end = '')

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

# objective function number 1
def getAmountOfEnabledPAs(PAs):
    numberOfPAsEnabled = 0
    for PA in PAs:
        if PA.enabled:
            numberOfPAsEnabled = numberOfPAsEnabled + 1
    return numberOfPAsEnabled

def getDistanceBetweenPAAndClient(PA, client):
    xsDistance = PA.x - client.x
    ysDistance = PA.y - client.y
    return math.sqrt( xsDistance**2 + ysDistance**2 )


# objective function number 2
def getTotalDistanceBewtweenEnablePAsAndClients(PAs, clients):
    totalDistance = 0
    for PA in PAs:
        for client in clients:
            distance = getDistanceBetweenPAAndClient(PA, client)

def main():
    clients = getClients()

    numberOfClients = len(clients)
    print(f"Number of clients: {numberOfClients}\n")
    
    PAs = factoryPAs()
    updatedPAs = initialConstructiveHeuristic(PAs, clients)

    #printPAs(updatedPAs)


main()