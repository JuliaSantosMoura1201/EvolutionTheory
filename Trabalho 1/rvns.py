"""
@author: Lucas S Batista
"""


'''
Importa os módulos usados
'''
import numpy as np
import copy


'''
Implementa a função shake
'''
def shake(x,k,probdata):
    
    y = copy.deepcopy(x)
    r = np.random.permutation(probdata.n)       
    
    if k == 1:             # apply not operator in one random position
        y.solution[r[0]] = not(y.solution[r[0]])
        
    elif k == 2:           # apply not operator in two random positions        
        y.solution[r[0]] = not(y.solution[r[0]])
        y.solution[r[1]] = not(y.solution[r[1]])
        
    elif k == 3:           # apply not operator in three random positions
        y.solution[r[0]] = not(y.solution[r[0]])
        y.solution[r[1]] = not(y.solution[r[1]])
        y.solution[r[2]] = not(y.solution[r[2]])        
    
    return y

'''
Implementa a função neighborhoodChange
'''
def neighborhoodChange(x, y, k):
    
    if y.single_objective_value < x.single_objective_value:
        x = copy.deepcopy(y)
        k = 1
    else:
        k += 1
        
    return x, k


'''
Implementa uma metaheurística RVNS
'''
def rvns_approach(fobj,x,probdata,approachinfo,maxeval=1000):
    
    # Contador do número de soluções candidatas avaliadas
    num_sol_avaliadas = 0

    # Máximo número de soluções candidatas avaliadas
    max_num_sol_avaliadas = maxeval

    # Número de estruturas de vizinhanças definidas
    kmax = 3
       
    # Avalia solução inicial
    x = fobj(x,approachinfo,probdata)
    num_sol_avaliadas += 1
    
    
    # Ciclo iterativo do método
    while num_sol_avaliadas < max_num_sol_avaliadas:

        k = 1
        while k <= kmax:

            # Gera uma solução candidata na k-ésima vizinhança de x          
            y = shake(x,k,probdata)
            y = fobj(y,approachinfo,probdata)
            num_sol_avaliadas += 1

            # Atualiza solução corrente e estrutura de vizinhança (se necessário)
            x, k = neighborhoodChange(x, y, k)
    
    return x


