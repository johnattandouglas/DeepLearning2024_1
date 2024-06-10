import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


caminho_pasta = os.getcwd()
dados = pd.read_csv(caminho_pasta + "/BaseCidade2019.csv", sep=";")

# Selecionando as features numéricas e categóricas
numericos = ['month', 'weekday', 'day', 'hour', 'temperature', 'r_temperature', 'wind', 'humidity', 'dew_point', 'pressure']
categoricos = ['season', 'workday']

# Aplicando normalização às features numéricas
scaler = StandardScaler()
dados_numericos = pd.DataFrame(scaler.fit_transform(dados[numericos]), columns=numericos)

# Aplicando LabelEncoder às features categóricas
encoder = LabelEncoder()
dados_categoricos = dados[categoricos].apply(encoder.fit_transform)

# Concatenando os resultados
processados = pd.concat([dados_numericos, dados_categoricos, dados['qtd']], axis=1)
df = processados

X = df.drop(columns=['qtd'])
y = df['qtd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X = df.drop(columns=['qtd'])
y = df['qtd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Função para calcular o RMSE ponderado
def weighted_rmse(y_true, y_pred, weight_factor):
    errors = y_true - y_pred
    weighted_errors = np.where(errors < 0, weight_factor * errors ** 2, errors ** 2)
    weighted_rmse = np.sqrt(np.mean(weighted_errors))
    return weighted_rmse

# Função para calcular o MAE ponderado
def weighted_mae(y_true, y_pred, weight_factor):
    errors = y_true - y_pred
    weighted_errors = np.where(errors < 0, weight_factor * np.abs(errors), np.abs(errors))
    weighted_mae = np.mean(weighted_errors)
    return weighted_mae

# Definindo os regressores
regressors = [
    # ('LR', LinearRegression()),
    # ('SVR', SVR()),
    # ('MLP', MLPRegressor()),
    ('DT', DecisionTreeRegressor()),
    ('RFR', RandomForestRegressor())
]

# Listas para armazenar os resultados
results_r2 = []
results_mae = []
results_maeP = []
results_rmse = []
results_rmseP = []
names = []


def evaluate_regressors(X_train, y_train, X_test, y_test, weight):
    for name, regressor in regressors:
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        maeP = weighted_mae(y_test, y_pred, weight)
        rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE = sqrt(MSE)
        rmseP = weighted_rmse(y_test, y_pred, weight)
        results_r2.append(r2)
        results_mae.append(mae)
        results_rmse.append(rmse)
        results_maeP.append(maeP)
        results_rmseP.append(rmseP)
        names.append(name)
        print(f"{name}: R² = {r2:.3f}, MAE = {mae:.3f}, MAE* = {maeP:.3f}, RMSE = {rmse:.3f}, RMSE* = {rmseP:.3f}")

# Chamada da função evaluate_regressors passando seus dados de treino e teste
weight_factor = 2 # Peso maior para erros negativos
evaluate_regressors(X_train, y_train, X_test, y_test, weight_factor)