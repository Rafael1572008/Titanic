# Predição de idade
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('regresao_linear_idade\\09 - Projeto - Regressao Linear - Predicao da Idade/titanic/train.csv')
test = pd.read_csv('regresao_linear_idade\\09 - Projeto - Regressao Linear - Predicao da Idade/titanic/test.csv')

train.set_index('PassengerId', inplace=True)
test.set_index('PassengerId', inplace=True)


# Verificar Shape dos dados
train.shape


# Concatenar train e teste
data = pd.concat([train, test], sort=False)

# Traduzir colunas
data.columns = ['Sobreviveu', 'Classe', 'Nome', 'Sexo', 'Idade', 'IrmaosConjuge', 'PaisFilhos', 'Bilhete',
       'Tarifa', 'Cabine', 'Embarque']

# Traduzir Sexo
data['Sexo'] = data['Sexo'].map({'male': 'homem', 'female': 'mulher'})


# Averiguar dados nulos
print(data.isnull().sum())


# Remover cabine
data.drop('Cabine', axis=1, inplace=True)
print(data.isnull().sum())

# Prencher embarque com a moda
mode = data['Embarque'].mode()[0]
print(mode)


# preencher dados nulos
data['Embarque'] = data['Embarque'].fillna(mode)
print(data.isnull().sum())
print()


# Preencher tarífa
mode = data['Tarifa'].mean()
data['Tarifa'] = data['Tarifa'].fillna(mode)
print(data.isnull().sum())
print()


# Verificar as correlação (apenas numéricas)
print(data.select_dtypes(include=['number']).corr())

# Visualizar com heaftMap
# sns.heatmap(data.select_dtypes(include=['number']).corr(), annot=True)
# plt.show()


# Remover coluna 'Sobreviver'. Uma vez que não muita relação com a idade e possui muitos dados nulos o ideal e retirar
data.drop('Sobreviveu', axis=1, inplace=True)

print(data.isnull().sum())


# Engenharia de recursos
data['Titulo'] = data['Nome'].str.extract('([a-zA-Z]+)\.')

data['Titulo'] = data['Titulo'].apply(lambda x: x if x in ['Miss', 'Master', 'Mr', 'Mrs'] else x)
print(data.head())

# traduzir Titulos
titulos_traduzidos = {
    'Master': 'Menino',
    'Miss': 'Solteira',
    'Mr': 'HomenAdulto',
    'Mrs' : 'Casada'
}

data['Titulo'] = data['Titulo'].map(titulos_traduzidos)
print(data.shape)