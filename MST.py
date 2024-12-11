import numpy as np
import heapq

def primMST(grafo):
    V = len(grafo)  
    pai = [-1] * V  
    chave = [float('inf')] * V  
    V_bool = [False] * V 

    chave[0] = 0
    min_heap = [(0, 0)] 

    while min_heap:
        
        _, u = heapq.heappop(min_heap)
        V_bool[u] = True

        
        for v in range(V):
            
            if grafo[u][v] > 0 and not V_bool[v] and chave[v] > grafo[u][v]:
                chave[v] = grafo[u][v]
                pai[v] = u
                heapq.heappush(min_heap, (chave[v], v))

    
    MST = [[0] * V for _ in range(V)]
    for v in range(1, V):
        u = pai[v]
        MST[u][v] = grafo[u][v]
        MST[v][u] = grafo[u][v]

    return MST

def MST(matriz_distancias, mpts):
    
    #print("inicializando MST", end="... ")

    # 1- Calcular a core distance
    core_distance = np.zeros(matriz_distancias.shape[0])
    for i in range(matriz_distancias.shape[0]):
        # Descobre os k indices os quais i vai ter aresta pra eles - incluo ele
        distancias_vizinhos = np.sort(matriz_distancias[i])[:mpts]
        #vizinhos = np.partition(matriz_distancias[i], mpts)[:mpts]
        core_distance[i] = distancias_vizinhos[-1]

    # 2- Criar grafo de Mutual Reachability Distance
    grafoMRD = np.zeros((matriz_distancias.shape[0],matriz_distancias.shape[1]))
    for i in range(matriz_distancias.shape[0]):
        for j in range(matriz_distancias.shape[1]):
            grafoMRD[i][j] = max(core_distance[i], core_distance[j], matriz_distancias[i][j])

    # 3- Gerar MST: Aplicar Prim
    MST = np.array(primMST(grafoMRD))

    MST[MST != 0] = 1

    #print("feito")

    return MST