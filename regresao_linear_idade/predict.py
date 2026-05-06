# Predição de idade
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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


#Não usar nome nem bilhete
data.drop(['Nome', 'Bilhete'], axis=1, inplace=True)


# Averiguar informações do DataFrame
print(data.info())

# Engenharia de recursos - 2
solterias_com_pais = data.loc[(data['Titulo'] == 'Solteira') & (data['PaisFilhos'] >= 1)]

# plt.hist(solterias_com_pais['Idade'], bins=15)
#plt.show()


# Média mulheeres solterias
m = data.loc[data['Titulo'] == 'Solteira']['Idade'].mean()

# Distribuição da idade de mulheres solteiras
# plt.hist(data.loc[data['Titulo'] == 'Solteira']['Idade'], bins=15)

# Distribuição da idade de mulheres casadas
# plt.hist(data.loc[data['Titulo'] == 'Casada']['Idade'], bins=15)
# plt.show()

# Engenharia de recursos - 3
data['Solteira_com_pais'] = 0

for idx, _ in data.iterrows():
    if idx in solterias_com_pais.index:
        data.loc[idx, 'Solteira_com_pais'] = 1

print(data.loc[data['Solteira_com_pais'] == 1]['Idade'].mean())
print(data.loc[data['Solteira_com_pais'] == 0]['Idade'].mean())


# Aplicação de Variavel Dummy
# Variaveis Dummy servem para transformar variáveis categóricas em numéricas, para que possam ser utilizadas em modelos de machine learning
data['Sexo'] = data['Sexo'].map({'homem': 0, 'mulher': 1})
print(data['Sexo'].head())

data = pd.get_dummies(data, columns=['Classe', 'Embarque', 'Titulo'], drop_first=True)
print(data.head())
print(data.shape)


# Definir X/e Train/Test

# Dados com idade conhecida → treino
train_idade = data.loc[data['Idade'].notnull()]

# Dados sem idade → predição
test_idade = data.loc[data['Idade'].isnull()]
print(train_idade.shape, test_idade.shape)

# definir X e y
x = train_idade.drop('Idade', axis=1)
y = train_idade['Idade']

# Instanciar o modelo de regressão linear
ln = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123) # Dividir os dados em treino e teste, 30% para teste e 70% para treino

ln.fit(x_train, y_train) # Treinar o modelo


pred = ln.predict(x_test) # Fazer previsões com o modelo treinado
print(pred)


# avaliar o modelo
from sklearn.metrics import mean_squared_error

# R²
score = ln.score(x_test, y_test) # R²
print(f'R²: {score}')


# RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f'RMSE: {rmse}')



# Visualizar residuos em scatter plot
# plt.scatter(y_test, pred, color='blue', s=40)
# plt.show()

# Aplicar modelo nos dados nulos
x_pred = test_idade.drop('Idade', axis=1)
pred_idade = ln.predict(x_pred)


test_idade = test_idade.copy()
test_idade['Idade'] = pred_idade
print(test_idade['Idade'].isnull().sum())


# Averiguar Shape
print(test_idade.shape, train_idade.shape)

idade = pd.concat([train_idade, test_idade], sort=False)


# Gerar CSV
idade_completa = pd.DataFrame({'IdPassgeiro': idade.index, 'Idade': idade['Idade']})
plt.scatter(idade_completa['IdPassgeiro'], idade_completa['Idade'], color='blue', s=40)
plt.show()

# csv
idade_completa.to_csv('index_idade.csv', index=False)

'''
1. juntar dados
2. limpar dados
3. criar features inteligentes
4. transformar em numérico
5. separar treino e predição
6. treinar modelo
7. avaliar modelo
8. prever dados faltantes
9. reconstruir dataset
10. exportar
'''