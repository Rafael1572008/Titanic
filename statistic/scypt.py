def pec(x):
    return "{:.2f}%".format(x * 100)

import pandas as pd
df = pd.read_csv('statistic\\import\\07 - Estatistica/weight-height.csv')


# Estiatística descritiva

## Obter homens apenas homens 
men = df[df['Gender'] == 'Male']

## Altura em cm
men['Height'] = men['Height'] * 2.54

## remover coluna de largura
cinco = men.drop(columns=['Weight', 'Gender']).head()

# Variância e Desvio Padrão

## Média
media = cinco['Height'].mean()
print(f'Média: {media}')

n = len(cinco)

## Variância
variancia = sum((cinco['Height'] - media) ** 2) / n
print(f'Variância: {variancia}')

# Desvio Padrão
desvio_padrao = variancia ** 0.5
print(f'Desvio Padrão: {desvio_padrao}')

# Coeficiente de Variação
cinco_women = df[df['Gender'] == 'Female'].drop(columns=['Weight', 'Gender']).head()

## Desvio Padrão manual das mulheres
desvio_pad = sum((cinco_women['Height'] - cinco_women['Height'].mean()) ** 2) / len(cinco_women)
desvio_pad = desvio_pad ** 0.5
print(f'Desvio Padrão das mulheres: {desvio_pad}')

## Média das mulheres
mean_women = cinco_women['Height'].mean()
print(f'Média das mulheres: {mean_women}')

## Coeficiente de Variação
coeficiente_variacao = desvio_pad / mean_women
print(f'Coeficiente de Variação: {pec(coeficiente_variacao)}')
# -----

## homens, coeficiente de variação
coeficiente_variacao_homens = desvio_padrao / media
print(f'Coeficiente de Variação dos homens: {pec(coeficiente_variacao_homens)}')

# Covariância
topcinco_men = men.head()
topcinco_men['Peso'] = topcinco_men['Weight'] * 0.453592

média_topcinco_men = topcinco_men['Height'].mean()

# culculo
covariancia = sum((topcinco_men['Height'] - média_topcinco_men) * (topcinco_men['Peso'] - topcinco_men['Peso'].mean())) / (len(topcinco_men) - 1)
print(f'Covariância: {covariancia}')

# Correlação
## Desvio padrão do peso
desvio_padrao_peso = sum((topcinco_men['Peso'] - topcinco_men['Peso'].mean()) ** 2) / (len(topcinco_men) - 1)
desvio_padrao_peso = desvio_padrao_peso ** 0.5

## Desvio padrão da altura
desvio_padrao_altura = sum((topcinco_men['Height'] - média_topcinco_men) ** 2) / (len(topcinco_men) - 1)
desvio_padrao_altura = desvio_padrao_altura ** 0.5

# Calculo
correlacao = covariancia / (desvio_padrao_peso * desvio_padrao_altura)
print(f'Correlação: {pec(correlacao)}')

# Biblioteca embutida no Pd
correlacao_biblioteca = topcinco_men['Height'].corr(topcinco_men['Peso'])
print(f'Correlação usando biblioteca: {pec(correlacao_biblioteca)}')