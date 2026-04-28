import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('statistic\\import\\07 - Estatistica/weight-height.csv')

# Amostra aleatória de 5
# np.random.seed(123)  # Definir uma semente para reprodutibilidade
# df = df['Weight'].sample(5)


# Gerar 50 médias amaostrais de 5 amostras cada
np.random.seed(123)  # Definir uma semente para reprodutibilidade
sample_means = []
for _ in range(200):
    amostra = df['Weight'].sample(5)
    media_amostral = amostra.mean()
    sample_means.append(media_amostral)


# plotar
plt.hist(sample_means, color='red', edgecolor='black')
plt.title('Médias Amostrais de Peso')
plt.xlabel('Amostra')
plt.show()


print(sample_means)