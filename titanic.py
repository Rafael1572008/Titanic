import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data =  pd.read_csv('train.csv')

## Renomear
data.columns = ['IdPassageiro', 'Sobreviveu', 'Classe', 'Nome', 'Sexo', 'Idade', 'IrmaosConjuge',
       'PaisFilhos', 'Bilhete', 'Tarifa', 'Cabine', 'Embarque']

data['Sexo'].replace({'male': 'homen',
                      'female': 'mulher'}, inplace=True)

## Primeira letra da cabine ou nulo
data['Cabine'] = data['Cabine'].apply(lambda x: x[0] if pd.notnull(x) else np.nan)

# print(data['Cabine'].head())


#Histograma
#plt.hist(data['Idade'].dropna())
#plt.title('Distribuição de idades')
#plt.ylabel('Pessoas')
#plt.xlabel('Idade')
#plt.show()

# print(data['Classe'].unique())

#plt.hist(data['Classe'])
#plt.title('Distribuição de idades')
#plt.ylabel('Pessoas')
#plt.xlabel('Idade')
#plt.show()

# Visualizar os dois graficos
# plt.figure(figsize=(10, 5)) # Tamanho da figura
# plt.subplot(2, 1, 1) #linha, colunas, plot1
# plt.hist(data['Idade'].dropna())
# plt.title('Distribuição das rendas') # Titulo


#plt.subplot(2, 1, 2) #linha, colunas, plot1
#plt.hist(data['Classe'])
#plt.title('Distribuição das classes')

#plt.tight_layout() # Tirar uma distancia

#plt.show()

# Subplot 2
# f, ax = plt.subplots(1, 2, figsize=(6, 3), dpi=150) # Uma linha, duas colunas, tamnho da imagem, qualidade
# ax[i] sendo i um inteiro representando o plot

# plot 1
# ax[0].hist(data['Idade'].dropna())
# ax[0].set_title('Distribuição das idades')

# Plot 2
# ax[1].hist(data['Classe'])
# ax[1].set_title('Distribuição das classes')
# plt.show()


# Seaborn
#Faz o matplotlib ser mais automatizado

print(data['Sobreviveu'].value_counts()) # Contar ocerrenci de valores

f, ax = plt.subplots(1, 2, figsize=(18, 5)) # Uma linha, duas colunas, tamnho da imagem
data['Sobreviveu'].value_counts().plot.pie(ax=ax[0])
plt.show()