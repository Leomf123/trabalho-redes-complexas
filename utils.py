from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

def normalizar_dados(nome_dado, dados):

    datasets_Normalizar = [
        "autoPrice.data",
        "banknote-authentication.data",
        "stock.data",
        "transplant.data"
    ]    

    dados_normalizados = np.array(dados)

    if nome_dado in datasets_Normalizar:
        mean = np.mean(dados, axis=0)
        std = np.std(dados, axis=0)
        dados_normalizados = (dados - mean) / std
    
    return dados_normalizados

def gerar_matriz_distancias(X, Y, medida_distancia ):

  matriz = cdist(X, Y, medida_distancia )

  return matriz

def definir_medida_distancia(nome_dado):

    datasets_euclidean = [
        "autoPrice.data",
        "banknote-authentication.data",
        "chscase_geyser1.data",
        "diggle_table.data",
        "iris.data",
        "seeds.data",
        "segmentation-normcols.data",
        "stock.data",
        "transplant.data",
        "wdbc.data",
        "wine-187.data",
        "yeast_Galactose.data",
        "mfeat-factors.data",
        "mfeat-karhunen.data",
        "cardiotocography.data"
    ]    
    datasets_tanimoto = [
        "ace_ECFP_4.data",
        "ace_ECFP_6.data",
        "cox2_ECFP_6.data",
        "dhfr_ECFP_4.data",
        "dhfr_ECFP_6.data",
        "fontaine_ECFP_4.data",
        "fontaine_ECFP_6.data",
        "m1_ECFP_4.data",
        "m1_ECFP_6.data",
    ]

    datasets_cosine = [
        "articles_1442_5.data",
        "articles_1442_80.data",
        "analcatdata_authorship-458.data",
        "armstrong2002v1.data",
        "chowdary2006.data",
        "gordon2002.data",
        "semeion.data"
    ]

    if nome_dado in datasets_tanimoto:
        return 'rogerstanimoto'
    elif nome_dado in datasets_cosine:
        return  'cosine'
    else:
        return 'euclidean'

def checar_matrix_adjacencias(matriz_adjacencias):

    simetrica = True
    conectado = True
    positivo = True
    for i in range(matriz_adjacencias.shape[0]):
        if np.sum(matriz_adjacencias[i]) == 0:
            conectado = False
        for j in range(matriz_adjacencias.shape[1]):
            if matriz_adjacencias[i][j] != matriz_adjacencias[j][i]:
                simetrica = False
    
    if (np.any(matriz_adjacencias < 0)):
        positivo = False

    return simetrica, conectado, positivo

def ordem_rotulos_primeiro(rotulos):

  #pegar posições existe rotulo
  posicoes_rotulos =  np.where( rotulos != 0)[0]

  # Reordenar as posições para os rotulados vim primeiro
  ordemObjetos = np.arange(rotulos.shape[0])
  # Retira dos indices os que são rotulados
  ordemObjetos = np.setdiff1d(ordemObjetos, posicoes_rotulos)
  # ordemObjetos será uma lista onde os indices dos objetos rotulados
  # vem primeiro, depois o resto em ordem crescente de indice
  ordemObjetos = np.concatenate((posicoes_rotulos,ordemObjetos))

  return posicoes_rotulos, ordemObjetos

def divisao_L(matriz_pesos):

    # Calculo da matriz diagonal: Uma matriz de grau de cada um dos vertices
    D = np.zeros(matriz_pesos.shape)
    np.fill_diagonal(D, np.sum(matriz_pesos, axis=1))
    # Calculo da matriz laplaciana
    L= 1.01*D - matriz_pesos

    # Calculo da laplaciana normalizada
    matriz_identidade = np.eye(matriz_pesos.shape[0])
    D_inv_raiz = np.diag(1 / np.sqrt(np.diag(D)))
    L_normalizada = 1.01*matriz_identidade - D_inv_raiz.dot(matriz_pesos).dot(D_inv_raiz)

    return L_normalizada

def gravar_resultados(test_ID, nome_dataset, k, adjacencia, simetrica, conectado, positivo, r, e, seed, nRotulos, acuracia, f_measure):

    if test_ID == 0: 

        # Criando um dataframe
        dados = [{'test_ID': test_ID, 'Dataset': nome_dataset, 'Adjacencia': adjacencia, 'k': k, 'Simetrica': simetrica, 'Conectado': conectado, 'Positivo': positivo, 'PorcRot': r, 'NumExp': e, 'SeedExp': seed, 'NumNRot': nRotulos, 'Acuracia': acuracia, 'F_measure': f_measure}]

        df = pd.DataFrame(dados)
        # salvo arquivo csv
        df.to_csv('ResultadosRC.csv', index=False)

    else:
        
        # leio arquivo csv existente e salvo df
        df = pd.read_csv('ResultadosRC.csv')
  
        # Adicionando dados
        dados = [{'test_ID': test_ID, 'Dataset': nome_dataset, 'Adjacencia': adjacencia, 'k': k, 'Simetrica': simetrica, 'Conectado': conectado, 'Positivo': positivo, 'PorcRot': r, 'NumExp': e, 'SeedExp': seed, 'NumNRot': nRotulos, 'Acuracia': acuracia, 'F_measure': f_measure}]

        dados = pd.DataFrame(dados)
        df = pd.concat([df, dados], ignore_index=True)

        # salvo arquivo csv mesmo lugar do outro
        df.to_csv('ResultadosRC.csv', index=False)
