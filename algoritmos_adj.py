from KNN import knn
from MST import MST

def gerar_matriz_adjacencias(dados, matriz_distancias, medida_distancia, k = 4, algoritmo = 'mutKNN'):
  
  if algoritmo in ['mutKNN', 'symKNN', 'symFKNN']:
    return knn(dados, matriz_distancias, medida_distancia, k, algoritmo)
  
  elif algoritmo == "MST":
    return MST(matriz_distancias, k)