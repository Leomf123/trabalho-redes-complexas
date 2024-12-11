import numpy as np

# LGC para calcular a matriz de rótulos propagados
# entrada: matriz de pesos, rótulos e parametro regularização
# saída: vetor de rótulos propagados
def LGC(L_normalizada, matriz_rotulos, ordemObjetos, posicoes_rotulos, rotulos, parametro_regularizacao):

  #print("inicializando LGC", end="... ")

  # Calculo da laplaciana normalizada
  matriz_identidade = np.eye(matriz_rotulos.shape[0])
 
  f = np.linalg.inv(matriz_identidade + L_normalizada/parametro_regularizacao).dot(matriz_rotulos)

  # Formatacao dos dados nao rotulados
  ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]

  resultado = np.array(rotulos) 
  for i in range(ordemNaoRotulado.shape[0]):
    rotulo = np.argmax(f[ordemNaoRotulado[i],:]) + 1
    resultado[ordemNaoRotulado[i]] = rotulo

  #print("feito")

  return resultado
