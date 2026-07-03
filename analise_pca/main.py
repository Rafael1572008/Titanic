# Principal componet Analysis (PCA)

# Queremos diminuir as quantidades de dimensões nos dados sem perder os dados

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA # decompor os dados
from sklearn.preprocessing import StandardScaler # Escalar os dados
from sklearn import datasets

# Iris
iris = datasets.load_iris()


x = iris['data']
y = iris['target']

# Fazera normalização
x = StandardScaler().fit_transform(x)


dataframe = pd.DataFrame(x, columns=iris['feature_names'])
print(dataframe)


# Implementação de PCA
pca = PCA(n_components=2) # diminuir as 4 dimensões para 2

pc = pca.fit_transform(x) # PC | Principal components

plt.scatter(pc[:,0], pc[:,1])
# plt.show()
plt.close()


plt.scatter(pc[y==0,0] , pc[y==0, 1], label='setosa')
plt.scatter(pc[y==1,0] , pc[y==1, 1], label='versiclour')
plt.scatter(pc[y==2,0] , pc[y==2, 1], label='verginica')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
# plt.show()
plt.close()

print(pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))


# Dígitos
digitos = datasets.load_digits()
x = digitos['data']
y = digitos['target']

pca = PCA(n_components=2)
x_reduzido = pca.fit_transform(x)

print(f'Projetando:  {x.shape[1]} Dimensões em apenas 2')
plt.figure(figsize=(12, 10))
plt.scatter(x_reduzido[:,0], x_reduzido[:,1], c = y, edgecolors='black', alpha=0.6, s=80)
# plt.show()
plt.close()

# TSNE | Mais detalhado e preciso que o PCA Obs: Gasta mais


# Reduzir para melhor dimensão

pca = PCA(0.95) # Porcentagem de integridade dos dados
pca.fit(x)

print(pca.n_components_) # Features para manter 90% de integridade
