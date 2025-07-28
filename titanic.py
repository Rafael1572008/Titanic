import pandas as pd
import numpy as np

data =  pd.read_csv('C:/Users/aline/Documents/Udemy/Prog/train.csv')

## Renomear
data.columns = ['IdPassageiro', 'Sobreviveu', 'Classe', 'Nome', 'Sexo', 'Idade', 'IrmaosConjuge',
       'PaisFilhos', 'Bilhete', 'Tarifa', 'Cabine', 'Embarque']

data['Sexo'].replace({'male': 'homen',
                      'female': 'mulher'}, inplace=True)

## Primeira letra da cabine ou nulo
data['Cabine'] = data['Cabine'].apply(lambda x: x[0] if pd.notnull(x) else np.nan)

print(data['Cabine'].head())