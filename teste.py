import numpy as np
import pandas as pd
import random

from algoritmos_adj import gerar_matriz_adjacencias
from processar_rotulos import retirar_rotulos, medidas_qualidade, one_hot
from LGC import LGC
from utils import ordem_rotulos_primeiro, divisao_L, gravar_resultados, definir_medida_distancia
from utils import normalizar_dados, checar_matrix_adjacencias, gerar_matriz_distancias

def teste(datasets, K, Adjacencia, Quantidade_rotulos, Quantidade_experimentos):

    test_ID = 0

    # 1 - Para cada dataset
    for nome_dataset in datasets:

        print("Dataset: ", nome_dataset)
        # Lendo dados
        df = pd.read_csv('data/' + nome_dataset, header=None)

        # Conversão para numpy
        dados = df.to_numpy()
        # Separando rótulos dos dados
        ultima_coluna = dados.shape[1] - 1
        rotulos = np.array(dados[:,ultima_coluna], dtype='int64')
        dados = np.array(dados[:,:ultima_coluna])
        # Pegar classes
        classes = np.unique(rotulos)

        # Normalizar dados
        dados = normalizar_dados(nome_dataset, dados)

        # medida_distancia = 'euclidean'
        medida_distancia = definir_medida_distancia(nome_dataset)
        matriz_distancias = gerar_matriz_distancias(dados, dados, medida_distancia)
            
        del df
        # 2 - Para cada valor de K
        for k in K:

            # 3 - Para cada algoritmo de adjacencia
            for adjacencia in Adjacencia:
                # Gerar matriz de adjacencia
                matriz_adjacencias = gerar_matriz_adjacencias(dados, matriz_distancias, medida_distancia, k, adjacencia)

                simetrica, conectado, positivo = checar_matrix_adjacencias(matriz_adjacencias)

                #del matriz_distancias, matriz_adjacencias
                # 5 - Para cada quantidade de rotulos
                for r in Quantidade_rotulos:

                    #Gerar os seeds
                    seeds = random.sample(range(1, 200), Quantidade_experimentos )

                    # 6 - Quantidade de experimentos
                    for e in range(Quantidade_experimentos):

                        # Retirar quantidade de rotulos
                        rotulos_semissupervisionado = retirar_rotulos(rotulos, r, classes, seeds[e])

                        posicoes_rotulos, ordemObjetos = ordem_rotulos_primeiro(rotulos_semissupervisionado)

                        # Extracao das submatrizes da matriz laplaciana
                        L_normalizada = divisao_L(matriz_adjacencias)
                            
                        matriz_rotulos = one_hot(rotulos_semissupervisionado)
                        
                        # Usado no LGC
                        parametro_regularizacao = 0.9
                        rotulos_propagados = LGC(L_normalizada, matriz_rotulos, ordemObjetos, posicoes_rotulos, rotulos, parametro_regularizacao)

                        # Usar medidas de qualidade
                        acuracia, f_measure, nRotulos = medidas_qualidade(posicoes_rotulos, ordemObjetos, rotulos, rotulos_propagados)


                        # gravar resultado em uma linha usando pandas
                        gravar_resultados(test_ID, nome_dataset, k, adjacencia, simetrica, conectado, positivo, r, e, seeds[e], nRotulos, acuracia, f_measure)

                        #print("test_ID: ", test_ID, ' ', nRotulos)

                        test_ID += 1
