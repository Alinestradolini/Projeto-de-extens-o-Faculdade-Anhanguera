#Sistema de Monitoramento
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

class SMGDS:
    def __init__(self):
        self.data = pd.DataFrame(columns=["Country", "Goal", "Indicator", "Value", "Timestamp"])

    def add_data(self, country, goal, indicator, value, timestamp):
        new_row = {"Country": country, "Goal": goal, "Indicator": indicator, "Value": value, "Timestamp": timestamp}
        self.data = self.data.append(new_row, ignore_index=True)

    def analyze_data(self):
        # Análise básica usando pandas
        summary_stats = self.data.groupby("Indicator")["Value"].describe()
        print("Resumo Estatístico:")
        print(summary_stats)

        # Visualização de dados
        self.plot_data()

        # Detecção de anomalias usando Isolation Forest
        self.detect_anomalies()

    def plot_data(self):
        # Gráfico de barras para visualizar os valores por indicador
        plt.figure(figsize=(10, 6))
        self.data.groupby("Indicator")["Value"].mean().plot(kind='bar', color='skyblue')
        plt.title('Média de Valores por Indicador')
        plt.ylabel('Média de Valores')
        plt.xlabel('Indicador')
        plt.xticks(rotation=45)
        plt.show()

    def detect_anomalies(self):
        # Isolation Forest para detecção de anomalias
        X = self.data[["Value"]]
        clf = IsolationForest(contamination=0.1)
        self.data["Anomaly"] = clf.fit_predict(X)

        # Exibindo pontos anômalos
        anomalies = self.data[self.data["Anomaly"] == -1]
        print("\nPontos Anômalos:")
        print(anomalies)

if __name__ == "__main__":
    smgds = SMGDS()
    
    # Adicione dados fictícios para teste
    smgds.add_data("Country1", "Goal1", "Indicator1", 50, "2023-09-04")
    smgds.add_data("Country2", "Goal2", "Indicator2", 75, "2023-09-04")
    smgds.add_data("Country3", "Goal1", "Indicator1", 90, "2023-09-04")
    smgds.add_data("Country4", "Goal2", "Indicator2", 30, "2023-09-04")
    
    # Execute a análise dos dados
    smgds.analyze_data()

