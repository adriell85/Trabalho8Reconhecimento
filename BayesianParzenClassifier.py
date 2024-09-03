import numpy as np
from scipy.stats import multivariate_normal


class BayesianParzenClassifier:
    def __init__(self, bandwidth=1.0, epsilon=1e-5):
        self.bandwidth = bandwidth
        self.classes = None
        self.class_priors = {}
        self.class_samples = {}
        self.epsilon = epsilon

    def parzenEstimate(self, x, samples):
        n, d = samples.shape
        volume = (self.bandwidth ** d) + self.epsilon
        kernel_sum = 0

        for sample in samples:
            kernel_value = np.exp(-0.5 * np.sum(((x - sample) / (self.bandwidth + self.epsilon)) ** 2))
            kernel_sum += kernel_value

        return kernel_sum / (n * np.sqrt((2 * np.pi) ** d) * volume)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        for cls in self.classes:
            class_samples = X[y == cls]
            self.class_samples[cls] = class_samples
            self.class_priors[cls] = class_samples.shape[0] / X.shape[0]



    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                likelihood = self.parzenEstimate(x, self.class_samples[cls])
                prior = self.class_priors[cls]
                posterior = likelihood * prior
                posteriors.append(posterior)

            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)