# Exemplo do Dado
import numpy as np
import matplotlib.pyplot as plt

## Média
mean = np.mean([1, 2, 3, 4, 5, 6])

## Valor Aleatório
rend = np.random.randint(1, 7)

## Jogaar o dado Aleatoriamente 6x
array = np.random.randint(1, 7, size=6)

## Valore com seed
amostra = np.random.RandomState(seed=123).randint(1, 7, size=40)

## Média amostral
media_amostral = np.mean(amostra)


## Prova do Teorema do Limite Central
# Gerar 1000 amostras de tamanho 40
np.random.seed(123)  # Definir a semente para reprodutibilidade
amostras = [np.mean(np.random.randint(1, 7, size=40)) for _ in range(1000)] # Gerar 1000 amostras de tamanho 40 e calcular a média de cada amostra
plt.hist(amostras)
plt.show() ## Distribuição Normal


## Média das Médias Amostrais
media_das_medias = np.mean(amostras)
print(f'Média das Médias Amostrais: {media_das_medias}')

# Mediana
mediana = np.median(amostras)
print(f'Mediana das Médias Amostrais: {mediana}')

# Moda
import statistics as stats
moda = stats.mode(amostras)
print(f'Moda das Médias Amostrais: {moda}')