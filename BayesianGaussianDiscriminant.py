import numpy as np
from plotGaussian import plotGaussianDistribution3d, dispersionDataBlindClass,dispersionDataByClass,plotDadosColridos,plotCovarianceMatrix



class GaussianDiscriminantAnalysis:
    def __init__(self, CovacianceType='lda'):
        self.CovacianceType = CovacianceType
        self.means = {}
        self.PriorProbs = {}
        self.covariances = {}
        self.SharedCovariances = None


    def fit(self, xtrain, ytrain,baseName,isruningTrain,iteration):
        xtrain = np.array(xtrain)
        classes = np.unique(ytrain)
        n_features = xtrain.shape[1]
        self.classes = np.unique(ytrain)

        for c in classes:
            _classSamples = xtrain[ytrain == c]
            self.means[c] = np.mean(_classSamples, axis=0)
            self.PriorProbs[c] = _classSamples.shape[0] / xtrain.shape[0]
            if self.CovacianceType == 'qda':
                cov = np.cov(_classSamples, rowvar=False) + np.eye(n_features) * 1e-5
                self.covariances[c] = cov
                plotCovarianceMatrix(cov, baseName, iteration,'Bayesian_{}'.format(self.CovacianceType))
            elif self.CovacianceType == 'lda':
                # LDA utiliza a covariância compartilhada
                cov = np.cov(_classSamples, rowvar=False) + np.eye(n_features) * 1e-5

                if self.SharedCovariances is None:
                    self.SharedCovariances = cov * self.PriorProbs[c]
                    plotCovarianceMatrix(self.SharedCovariances, baseName, iteration, 'Bayesian_{}'.format(self.CovacianceType))
                else:
                    self.SharedCovariances += cov * self.PriorProbs[c]
                    plotCovarianceMatrix(self.SharedCovariances, baseName, iteration,
                                         'Bayesian_{}'.format(self.CovacianceType))
        if self.CovacianceType == 'lda':
            self.SharedCovariances /= len(classes)

        if (isruningTrain and self.CovacianceType == 'qda'):
            # dispersionDataBlindClass(xtrain, baseName, iteration, True,'Bayesian_{}'.format(self.CovacianceType))
            plotDadosColridos(xtrain, ytrain, baseName, iteration, True,'Bayesian_{}'.format(self.CovacianceType))
            # plotGaussianDistribution3d(self.CovacianceType,baseName, iteration, self.means, self.covariances, self.classes,featureIndices=(1, 2))
            # fileName = "DadosGaussiana/Dados_Plotagem_Gaussiana{}_base_{}_iteracao_{}.txt".format(baseName, baseName,iteration)
            # with open(fileName, 'w') as arquivo:
            #     arquivo.write("Dados de Treino.\n\n{}\n".format(xtrain))
    def predict(self, xtest,ytest,baseName,iteration,isRuningZ):
        if (isRuningZ == False):
            # dispersionDataBlindClass(xtest, baseName, iteration, False,'Bayesian_{}'.format(self.CovacianceType))
            plotDadosColridos(xtest, ytest, baseName, iteration, False, 'Bayesian_{}'.format(self.CovacianceType))
        # fileName = "DadosGaussiana/Dados_Plotagem_Gaussiana{}_base_{}_iteracao_{}.txt".format(baseName, baseName,
        #                                                                                       iteration)
        # with open(fileName, 'a') as arquivo:
        #     arquivo.write("Dados de Teste.\n\n{}\n".format(xtest))
        #     arquivo.write('\nIteração: {} :::::::::::::::::\n'.format(iteration))
        preds = []
        for sample in xtest:
            probs = {}
            for c in self.means:
                probs[c] = self._logPdf(sample, self.means[c], self.covariances[
                    c] if self.CovacianceType == 'qda' else self.SharedCovariances) + np.log(self.PriorProbs[c])
            preds.append(max(probs, key=probs.get))
        return np.array(preds)

    def _logPdf(self, sample, mean, covariance):
        size = len(mean)
        detCov = np.linalg.det(covariance)
        invCov = np.linalg.inv(covariance)
        factor = -0.5 * (size * np.log(2 * np.pi) + np.log(detCov))
        return factor - 0.5 * np.dot(np.dot((sample - mean).T, invCov), (sample - mean))



