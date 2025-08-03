import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data =  pd.read_csv('train.csv')

## Renomear
data.columns = ['IdPassageiro', 'Sobreviveu', 'Classe', 'Nome', 'Sexo', 'Idade', 'IrmaosConjuge',
       'PaisFilhos', 'Bilhete', 'Tarifa', 'Cabine', 'Embarque']

data['Sexo'].replace({'male': 'homen',
                      'female': 'mulher'}, inplace=True)

## Primeira letra da cabine ou nulo
data['Cabine'] = data['Cabine'].apply(lambda x: x[0] if pd.notnull(x) else np.nan)

print(data['Cabine'].head())


#Histograma
#plt.hist(data['Idade'].dropna())
#plt.title('Distribuição de idades')
#plt.ylabel('Pessoas')
#plt.xlabel('Idade')
#plt.show()

print(data['Classe'].unique())

plt.hist(data['Classe'])
plt.title('Distribuição de idades')
plt.ylabel('Pessoas')
plt.xlabel('Idade')
plt.show()

# Visualizar os dois graficos
plt.subplot(2, 1, 1) #linha, colunas, plot1
plt.hist(data['Idade'].dropna())
plt.title('Distribuição das rendas')


plt.subplot(2, 1, 2) #linha, colunas, plot1
plt.hist(data['Classe'])
plt.title('Distribuição das classes')

plt.tight_layout() # Tirar uma distancia

plt.show()