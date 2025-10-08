# ===============================================
# ANÁLISE EXPLORATÓRIA DO TITANIC
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================
# 1. Carregar e preparar os dados
# ===============================================

data = pd.read_csv('train.csv')

## Renomear as colunas para português
data.columns = [
    'IdPassageiro', 'Sobreviveu', 'Classe', 'Nome', 'Sexo', 'Idade',
    'IrmaosConjuge', 'PaisFilhos', 'Bilhete', 'Tarifa', 'Cabine', 'Embarque'
]

## Traduzir os valores da coluna Sexo
data['Sexo'].replace({'male': 'homen', 'female': 'mulher'}, inplace=True)

## Pegar apenas a primeira letra da cabine (ex: 'C85' → 'C') ou deixar nulo se for NaN
data['Cabine'] = data['Cabine'].apply(lambda x: x[0] if pd.notnull(x) else np.nan)

# print(data['Cabine'].head())  # Exemplo de visualização

# ===============================================
# 2. Exemplos de gráficos com Matplotlib
# ===============================================

# --- Histograma simples de idades ---
# plt.hist(data['Idade'].dropna())
# plt.title('Distribuição de idades')
# plt.ylabel('Pessoas')
# plt.xlabel('Idade')
# plt.show()

# print(data['Classe'].unique())  # Ver classes únicas

# --- Histograma simples de classes ---
# plt.hist(data['Classe'])
# plt.title('Distribuição das classes')
# plt.ylabel('Pessoas')
# plt.xlabel('Classe')
# plt.show()

# --- Dois gráficos na mesma figura (Subplots) ---
# plt.figure(figsize=(10, 5))  # Define o tamanho da figura
# plt.subplot(2, 1, 1)         # Linha, coluna, índice (1º gráfico)
# plt.hist(data['Idade'].dropna())
# plt.title('Distribuição das idades')

# plt.subplot(2, 1, 2)         # Linha, coluna, índice (2º gráfico)
# plt.hist(data['Classe'])
# plt.title('Distribuição das classes')

# plt.tight_layout()           # Ajusta o espaçamento entre gráficos
# plt.show()

# --- Subplots com f e ax (maneira mais profissional) ---
# f, ax = plt.subplots(1, 2, figsize=(6, 3), dpi=150)  # 1 linha, 2 colunas

# ax[0].hist(data['Idade'].dropna())
# ax[0].set_title('Distribuição das idades')

# ax[1].hist(data['Classe'])
# ax[1].set_title('Distribuição das classes')

# plt.show()

# ===============================================
# 3. Usando Seaborn (visualização mais automática)
# ===============================================

print(data['Sobreviveu'].value_counts())  # Conta quantos sobreviveram e quantos não

# --- Comparação de sobrevivência com Matplotlib ---
# f, ax = plt.subplots(1, 2, figsize=(18, 5))
# data['Classe'].value_counts().plot.pie(
#     ax=ax[0],
#     explode=[0.02, 0.02, 0.02],
#     autopct='%0.2f%%'
# )
# ax[0].set_ylabel('')  # Remove título do eixo Y

# --- Usando Seaborn para contar ---
# sns.countplot(x='Sobreviveu', data=data, ax=ax[1])
# ax[1].set_ylabel('')

# ===============================================
# 4. Contagem com separação por Sexo
# ===============================================

# sns.countplot(x='Sobreviveu', hue='Sexo', data=data)  # hue adiciona a divisão por cor (legenda)
# plt.show()

# ===============================================
# 5. Boxplot – Idade x Classe x Sexo
# ===============================================

# plt.figure(figsize=(5, 3), dpi=100)
# sns.boxplot(x='Classe', y='Idade', hue='Sexo', data=data)
# Mostra onde a maioria das idades está concentrada por classe e sexo

# ===============================================
# 6. Violinplot – mesma ideia, mas mais detalhado
# ===============================================

# sns.violinplot(x='Classe', y='Idade', hue='Sexo', data=data)
# Representa densidade (espessura indica concentração de valores)

# ===============================================
# 7. FacetGrid com Histograma (Idade x Sobrevivência)
# ===============================================

# g = sns.FacetGrid(data, col='Sobreviveu')  # Cria um gráfico para cada valor de 'Sobreviveu'
# g.map(plt.hist, 'Idade', bins=18)          # Mostra distribuição de idades em cada grupo

# ===============================================
# 8. FacetGrid com Barplot (Tarifa x Sexo x Embarque)
# ===============================================

# g = sns.FacetGrid(data, row='Embarque', col='Sobreviveu')
# 3 portos de embarque (S, C, Q) × 2 resultados (0, 1)
# g.map(sns.barplot, 'Sexo', 'Tarifa', alpha=0.5, ci=None)
# alpha controla transparência, ci=None remove intervalo de confiança

## Mudar tamanho da imagem
# fig = plt.gcf()
# fig.set_size_inches(10, 6)

# ===============================================
# 9. FacetGrid com Displot (Idade x Classe x Sobreviveu)
# ===============================================

# g = sns.FacetGrid(data, col='Sobreviveu', row='Classe', height=1.8, aspect=2.2)
# g.map(sns.histplot, 'Idade', bins=20)

# ===============================================
# 10. Catplot – ponto médio de sobrevivência por embarque
# ===============================================

# sns.catplot(x='Embarque', y='Sobreviveu', data=data, kind='point')
# Mostra a taxa média de sobrevivência em cada porto
# fig = plt.gcf()
# fig.set_size_inches(6, 3)
# plt.show()

# ===============================================
# 11. FacetGrid com Pointplot (Classe x Sobreviveu x Sexo, separado por Embarque)
# ===============================================

# Este gráfico mostra a taxa de sobrevivência média por Classe e Sexo,
# separada por porto de embarque.

# g = sns.FacetGrid(data, row='Embarque', height=1.8, aspect=2.2)
# g.map(sns.pointplot, 'Classe', 'Sobreviveu', 'Sexo')
# sns.pointplot faz uma média de sobrevivência (y=0~1)
# Classe no eixo x, sobrevivência no y, separado por Sexo
# Cada linha (row) representa um porto de embarque diferente (S, C, Q)
# g.add_legend()  # Adiciona legenda
# plt.show()

# ===============================================
# 12. Heatmap de Dados Ausentes
# ===============================================

# Este gráfico mostra onde há dados ausentes no DataFrame.

# print(data.isnull().sum())  # Conta valores nulos por coluna
# sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='magma') # Mostra onde há dados ausentes por um gráfico de calor
# plt.show() # Mostrar o gráfico

# ===============================================
# 13. Relpot – Matriz de Correlação
# ===============================================

# Este gráfico mostra a correlação entre variáveis numéricas.

# sns.relplot(x='Tarifa', y='Idade', hue='Sexo', data=data, size='Tarifa', sizes=(40, 400), alpha=0.7, palette='magma') # Relaciona Tarifa e Idade, separado por Sexo, tamanho do ponto indica valor da Tarifa
# plt.show()


# ===============================================
# 14. Scatterplot – Correlação simples
# ===============================================

# Este gráfico mostra a correlação entre Idade e Tarifa, separado por Embarque.

# sns.scatterplot(x='Idade', y='Tarifa', hue='Embarque', size='Idade', palette='Greens', data=data, sizes=(10, 200)) # Relaciona Idade e Tarifa, separado por Embarque, tamanho do ponto indica valor da Idade
# plt.show()

# ===============================================
# 15. Pairplot – Matriz de Correlação Completa
# ===============================================

#

sns.pairplot(data[['Tarifa', 'Idade', 'Classe', 'Sexo', 'Embarque']], hue='Classe') 
plt.show()