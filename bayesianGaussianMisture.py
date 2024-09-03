import numpy as np
import os

class BayesianGaussianMixtureClassifier:
    def __init__(self, n_components=1, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.models = {}
        self.priors = {}
        self.regularization = 1e-6

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        np.random.seed(42)


        means = X[np.random.choice(n_samples, self.n_components, replace=False)]

        covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])

        mixing_coeffs = np.ones(self.n_components) / self.n_components

        return means, covariances, mixing_coeffs

    def _e_step(self, X, means, covariances, mixing_coeffs):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            resp = self._multivariate_gaussian(X, means[k], covariances[k])
            responsibilities[:, k] = mixing_coeffs[k] * resp

        sum_responsibilities = responsibilities.sum(axis=1)[:, np.newaxis]
        responsibilities /= sum_responsibilities

        return responsibilities

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        means = np.zeros((self.n_components, n_features))
        covariances = np.zeros((self.n_components, n_features, n_features))
        mixing_coeffs = np.zeros(self.n_components)

        for k in range(self.n_components):
            resp = responsibilities[:, k]
            total_resp = resp.sum()


            means[k] = (X * resp[:, np.newaxis]).sum(axis=0) / total_resp


            diff = X - means[k]
            covariances[k] = np.dot(resp * diff.T, diff) / total_resp
            covariances[k] += np.eye(n_features) * self.regularization


            mixing_coeffs[k] = total_resp / n_samples

        return means, covariances, mixing_coeffs

    def _log_likelihood(self, X, means, covariances, mixing_coeffs):
        n_samples = X.shape[0]
        log_likelihood = 0

        for k in range(self.n_components):
            log_likelihood += mixing_coeffs[k] * self._multivariate_gaussian(X, means[k], covariances[k])

        return np.sum(np.log(log_likelihood + 1e-10))

    def _multivariate_gaussian(self, X, mean, covariance):
        n_features = X.shape[1]
        diff = X - mean
        inv_covariance = np.linalg.inv(covariance)
        exp_term = np.exp(-0.5 * np.sum(diff @ inv_covariance * diff, axis=1))
        norm_const = np.sqrt((2 * np.pi) ** n_features * np.linalg.det(covariance))

        return exp_term / (norm_const + 1e-10)

    def fit(self, X, y, baseName='', isruningTrain=False, iteration=0):
        self.models = {}
        self.priors = {}

        X = np.array(X)
        y = np.array(y)
        classes = np.unique(y)

        for c in classes:
            X_c = X[y == c]
            means, covariances, mixing_coeffs = self._initialize_parameters(X_c)

            log_likelihood_old = -np.inf
            for _ in range(self.max_iter):
                responsibilities = self._e_step(X_c, means, covariances, mixing_coeffs)
                means, covariances, mixing_coeffs = self._m_step(X_c, responsibilities)

                log_likelihood_new = self._log_likelihood(X_c, means, covariances, mixing_coeffs)
                if np.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                    break
                log_likelihood_old = log_likelihood_new

            self.models[c] = (means, covariances, mixing_coeffs)
            self.priors[c] = X_c.shape[0] / X.shape[0]

        if isruningTrain:
            self._plot_clusters(X, y, baseName, iteration, 'Train')
            fileName = f"Resultados_BayesianGaussianMixture/{baseName}/Dados_Plotagem_Bayesian_{baseName}_iteracao_{iteration}.txt"
            os.makedirs(os.path.dirname(fileName), exist_ok=True)
            with open(fileName, 'w') as arquivo:
                arquivo.write(f"Dados de Treino com Mixture of Gaussians\n\n")
                arquivo.write(f"{X}\n")

    def predict(self, X, baseName='', iteration=0, isRuningZ=False):
        X = np.array(X)
        predictions = []

        for x in X:
            posteriors = {}
            for c, (means, covariances, mixing_coeffs) in self.models.items():
                likelihood = np.sum([
                    mixing_coeffs[k] * self._multivariate_gaussian(x.reshape(1, -1), means[k], covariances[k])
                    for k in range(self.n_components)
                ])
                posteriors[c] = self.priors[c] * likelihood

            predictions.append(max(posteriors, key=posteriors.get))

        if not isRuningZ:
            self._plot_clusters(X, predictions, baseName, iteration, 'Test')
            fileName = f"Resultados_BayesianGaussianMixture/{baseName}/Dados_Plotagem_Bayesian_{baseName}_iteracao_{iteration}.txt"
            with open(fileName, 'a') as arquivo:
                arquivo.write(f"Dados de Teste com Mixture of Gaussians\n\n")
                arquivo.write(f"{X}\n")
                arquivo.write(f'\nIteração: {iteration} :::::::::::::::::\n')

        return np.array(predictions)

    def _plot_clusters(self, X, labels, baseName, iteration, phase):
        pass

    def evaluate(self, X, y):
        y_pred = self.predict(X, isRuningZ=True)
        accuracy = np.mean(y_pred == y)
        return accuracy, y_pred


