import numpy as np
from sklearn.neighbors import kneighbors_graph

# KNN para calcular a matriz de adjacencias
# entrada: matriz de distancia, k e tipo
# saida: matriz de adjacencia
def knn(dados, matriz_distancia, medida_distancia, k, tipo):

  # Converte em bool para usar a rogerstanimoto sem da warning
  if medida_distancia == 'rogerstanimoto':
     dados = dados.astype(bool)

  #print("inicializando " + tipo, end="... ")
  matriz_adjacencia =  kneighbors_graph(dados, k, mode='connectivity',  metric = medida_distancia, include_self=False).toarray()

  matriz_adjacencia_transposta = matriz_adjacencia.T

  if tipo == 'mutKNN':
     
    matriz_adjacencia = np.minimum(matriz_adjacencia, matriz_adjacencia_transposta)

    for i in range(matriz_adjacencia.shape[0]):
      # checar se ta isolado
      if np.sum(matriz_adjacencia[i]) == 0:
        k_indices = np.argsort(matriz_distancia[i])[:2]
        #k_indices = np.argpartition(matriz_distancia[i], 2)[:2]
        for k in k_indices:
          if i != k:
            matriz_adjacencia[i][k] = 1
            matriz_adjacencia[k][i] = 1

  elif tipo == 'symKNN':
    matriz_adjacencia = np.maximum(matriz_adjacencia, matriz_adjacencia_transposta)

  elif tipo == 'symFKNN':
    matriz_adjacencia = matriz_adjacencia + matriz_adjacencia_transposta
  
  return matriz_adjacencia
