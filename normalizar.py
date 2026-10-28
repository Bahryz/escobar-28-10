import pandas as pd
from sklearn import preprocessing
from pickle import dump, load


dados = pd.read_csv('dados_normalizar.csv')
print(dados.head(3))
# Separando colunas categoricas de numericas

dados_num = dados.drop(columns=['sexo'])
dados_cat = dados['sexo']

# Instanciando o normalizador

normalizador = preprocessing.MinMaxScaler()

# Treinar o modelo normalizador a partir dos dados numéricos 

modelo_normalizador = normalizador.fit(dados_num)

# Salvar o modelo normalizador para uso posterior

dump(modelo_normalizador, open('modelo_normalizador.model', 'wb'))

# Normalizar os dados numéricos
dados_num_normalizados = modelo_normalizador.fit_transform(dados_num)

print(dados_num_normalizados[:3])

#recompor o dataframe com os dados normalizados e a coluna categórica, atribuindo
dados_num_normalizados = pd.DataFrame(dados_num_normalizados, columns=dados_num.columns)


#Normalizar uma instancia só
#Nova instancia
instancia = dados_num.iloc[14]

#Normalizar a nova instancia usando o modlo normalizador salvo
instancia_normalizada = modelo_normalizador.transform([instancia])
print(instancia_normalizada)

instancia_natural = modelo.normalizador.inverse_transform(instancia_normalizada)
print(instancia_natural)