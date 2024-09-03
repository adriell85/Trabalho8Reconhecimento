import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



class BayesianGaussianRejectionQuant:
    def __init__(self, Wr=0.1):
        self.Wr = Wr
        self.means = {}
        self.priors = {}
        self.covariances = {}

    def fit(self, X, y, baseName='', isruningTrain=False, iteration=0):
        X = np.array(X)
        y = np.array(y)
        classes = np.unique(y)
        n_features = X.shape[1]

        for c in classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.covariances[c] = np.cov(X_c, rowvar=False) + np.eye(n_features) * 1e-5

        if isruningTrain:
            self._plot_clusters(X, y, baseName, iteration, 'Train')
            fileName = f"Resultados_BayesianGaussianRejectionQuant/{baseName}/Dados_Plotagem_Bayesian_{baseName}_Wr_{self.Wr}_iteracao_{iteration}.txt"
            os.makedirs(os.path.dirname(fileName), exist_ok=True)
            with open(fileName, 'w') as arquivo:
                arquivo.write(f"Dados de Treino com Wr={self.Wr}\n\n")
                arquivo.write(f"{X}\n")

    def predict(self, X, baseName='', iteration=0, isRuningZ=False):
        X = np.array(X)
        predictions = []
        rejection_decisions = []

        for x in X:
            posteriors = {}
            for c in self.means:
                mean = self.means[c]
                cov = self.covariances[c]
                prior = self.priors[c]
                likelihood = multivariate_normal.pdf(x, mean, cov)
                posteriors[c] = prior * likelihood

            sorted_posteriors = sorted(posteriors.items(), key=lambda item: item[1], reverse=True)
            max_posterior = sorted_posteriors[0][1]
            second_max_posterior = sorted_posteriors[1][1] if len(sorted_posteriors) > 1 else 0


            if (max_posterior - second_max_posterior) < self.Wr:
                predictions.append(-1)
                rejection_decisions.append(True)
            else:
                predictions.append(sorted_posteriors[0][0])
                rejection_decisions.append(False)

        if not isRuningZ:
            self._plot_clusters(X, predictions, baseName, iteration, 'Test')
            fileName = f"Resultados_BayesianGaussianRejectionQuant/{baseName}/Dados_Plotagem_Bayesian_{baseName}_Wr_{self.Wr}_iteracao_{iteration}.txt"
            with open(fileName, 'a') as arquivo:
                arquivo.write(f"Dados de Teste com Wr={self.Wr}\n\n")
                arquivo.write(f"{X}\n")
                arquivo.write(f'\nIteração: {iteration} :::::::::::::::::\n')


            accuracy = np.mean(np.array(predictions) == np.array(predictions))
            with open(f"Resultados_BayesianGaussianRejectionQuant/{baseName}/DadosRuns_{baseName}.txt", 'a') as arquivo:
                arquivo.write(f"Iteração {iteration}: Acurácia = {accuracy:.4f}\n")

        return np.array(predictions), np.array(rejection_decisions)

    def evaluate(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_pred, rejections = self.predict(X, isRuningZ=True)
        valid_indices = y_pred != -1

        if valid_indices.sum() == 0:
            accuracy = 0.0
        else:
            accuracy = np.mean(y_pred[valid_indices] == y[valid_indices])

        rejection_rate = np.mean(rejections)
        return accuracy, rejection_rate, y_pred

    def _plot_clusters(self, X, labels, baseName, iteration, phase):
        n_features = X.shape[1]
        for i in range(n_features):
            for j in range(i + 1, n_features):
                feature_1, feature_2 = i, j

                plt.figure()
                plt.scatter(X[:, feature_1], X[:, feature_2], c=labels, s=50, cmap='viridis')

                x_min, x_max = X[:, feature_1].min() - 1, X[:, feature_1].max() + 1
                y_min, y_max = X[:, feature_2].min() - 1, X[:, feature_2].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                     np.arange(y_min, y_max, 0.1))

                grid_points = np.c_[xx.ravel(), yy.ravel()]
                grid_points_expanded = np.zeros((grid_points.shape[0], n_features))
                grid_points_expanded[:, feature_1] = grid_points[:, 0]
                grid_points_expanded[:, feature_2] = grid_points[:, 1]

                Z, _ = self.predict(grid_points_expanded, isRuningZ=True)
                Z = Z.reshape(xx.shape)

                plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

                plt.title(
                    f'Bayesian Gaussian Rejection ({phase}) - Iteration {iteration} (Features {feature_1 + 1} vs {feature_2 + 1})')
                plt.xlabel(f'Feature {feature_1 + 1}')
                plt.ylabel(f'Feature {feature_2 + 1}')
                os.makedirs(f"Resultados_BayesianGaussianRejectionQuant/{baseName}", exist_ok=True)
                plt.savefig(
                    f"Resultados_BayesianGaussianRejectionQuant/{baseName}/Bayesian_{baseName}_{phase}_Wr_{self.Wr}_Iteration_{iteration}_Features_{feature_1 + 1}_vs_{feature_2 + 1}.png")
                plt.close()