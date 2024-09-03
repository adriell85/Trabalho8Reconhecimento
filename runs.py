import numpy as np
from KNN import KNN
from DMC import DMC
from BayesianClassifierWithRejection import BayesianGaussianRejectionQuant
from openDatasets import openIrisDataset, openDermatologyDataset,openBreastDataset,openColumnDataset,openArtificialDataset,datasetSplitTrainTest, openIrisDatasetRejectRun, openColumnDatasetRejectRun, openArtificialDatasetRejectRun
from plots import confusionMatrix, plotConfusionMatrix,plotDecisionSurface, plotAccuracyRejectionCurvePerWr,plotDataPoints
from bayesianGaussianMisture import BayesianGaussianMixtureClassifier
from BayesianParzenClassifier import BayesianParzenClassifier

def KNNRuns(base):
    convertRun = {
        0: openIrisDataset(),
        1: openColumnDataset(),
        2: openArtificialDataset(),
        3: openBreastDataset(),
        4: openDermatologyDataset()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial',
        3: 'Breast',
        4: 'Dermatology'

    }

    out = convertRun[base]
    x = out[0]
    y = out[1]
    originalLabels = out[2]
    accuracyList = []

    fileName = "DadosRuns/KNNRuns_{}.txt".format(convertDocName[base])
    with open(fileName, 'w') as arquivo:
        arquivo.write("Execução Iterações KNN {}.\n\n".format(convertDocName[base]))
        for i in range(20):
            print('\nIteração {}\n'.format(i))
            xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x, y, 80,'KNN',convertDocName[base])
            ypredict = KNN(xtrain, ytrain, xtest, 5)
            confMatrix = confusionMatrix(ytest, ypredict)
            print('Confusion Matrix:\n', confMatrix)
            plotConfusionMatrix(confMatrix,originalLabels,'KNN',i,convertDocName[base])
            accuracy = np.trace(confMatrix) / np.sum(confMatrix)
            print('ACC:', accuracy)
            arquivo.write("ACC: {}\n".format(accuracy))
            arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
            accuracyList.append(i)
            plotDecisionSurface(xtrain, ytrain,'KNN',i,convertDocName[base])
        print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
        arquivo.write(
            '\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))


def DMCRuns(base):
    convertRun = {
        0: openIrisDataset(),
        1: openColumnDataset(),
        2: openArtificialDataset(),
        3: openBreastDataset(),
        4: openDermatologyDataset()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial',
        3: 'Breast',
        4: 'Dermatology'

    }

    out = convertRun[base]
    x = out[0]
    y = out[1]
    originalLabels = out[2]
    accuracyList = []

    fileName = "DadosRuns/DMCRuns_{}.txt".format(convertDocName[base])
    with open(fileName, 'w') as arquivo:
        arquivo.write("Execução Iterações DMC.\n\n")
        for i in range(20):
            print('\nIteração {}\n'.format(i))
            xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x, y, 80,'DMC',convertDocName[base])
            ypredict = DMC(xtrain, ytrain, xtest)
            confMatrix = confusionMatrix(ytest, ypredict)
            print('Confusion Matrix:\n', confMatrix)
            plotConfusionMatrix(confMatrix,originalLabels,'DMC',i,convertDocName[base])
            accuracy = np.trace(confMatrix) / np.sum(confMatrix)
            print('ACC:', accuracy)
            arquivo.write("ACC: {}\n".format(accuracy))
            arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
            accuracyList.append(i)
            plotDecisionSurface(xtrain, ytrain,'DMC',i,convertDocName[base])
        print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
        arquivo.write(
            '\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))

def BayesianGaussianDiscriminantRuns(base):
    from BayesianGaussianDiscriminant import GaussianDiscriminantAnalysis
    convertRun = {
        0: openIrisDataset(),
        1: openColumnDataset(),
        2: openArtificialDataset(),
        3: openBreastDataset(),
        4: openDermatologyDataset()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial',
        3: 'Breast',
        4: 'Dermatology'

    }
    classifierMode = [
        'lda',
        'qda'
    ]

    for mode in classifierMode:
        out = convertRun[base]
        x = out[0]
        y = out[1]
        originalLabels = out[2]
        accuracyList = []
        fileName = "DadosRuns/BayesianRuns_{}_{}.txt".format(convertDocName[base],mode)
        with open(fileName, 'w') as arquivo:
            arquivo.write("Execução Iterações Bayesian {} {}.\n\n".format(convertDocName[base],mode))
            for i in range(21):
                print('\nIteração {}\n'.format(i))
                xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x, y, 80)
                model = GaussianDiscriminantAnalysis(mode)
                model.fit(xtrain, ytrain,convertDocName[base],True,i)
                ypredict = model.predict(xtest,ytest,convertDocName[base],i,False)
                confMatrix = confusionMatrix(ytest, ypredict)
                print('Confusion Matrix:\n', confMatrix)
                plotConfusionMatrix(confMatrix,originalLabels,'Bayesian_{}'.format(mode),i,convertDocName[base],)
                accuracy = np.trace(confMatrix) / np.sum(confMatrix)
                print('ACC:', accuracy)
                arquivo.write("ACC: {}\n".format(accuracy))
                arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
                accuracyList.append(accuracy)
                plotDecisionSurface(xtrain, ytrain,'Bayesian_{}'.format(mode),i,convertDocName[base])
            print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
            arquivo.write(
                '\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))




def KmeansQuantRuns(base):
    convertRun = {
        0: openIrisDataset(),
        1: openColumnDataset(),
        2: openArtificialDataset(),
        3: openBreastDataset(),
        4: openDermatologyDataset()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial',
        3: 'Breast',
        4: 'Dermatology'

    }


    out = convertRun[base]
    x = out[0]
    y = out[1]
    originalLabels = out[2]
    accuracyList = []
    fileName = "DadosRuns/kMeansQuant_{}.txt".format(convertDocName[base])
    with open(fileName, 'w') as arquivo:
        arquivo.write("Execução Iterações Kmeans {}.\n\n".format(convertDocName[base]))
        for i in range(21):
            print('\nIteração {}\n'.format(i))
            xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x, y, 80)
            model = KMeansQuant()
            model.fit(X=xtrain,y=ytrain,baseName=convertDocName[base],isruningTrain=True,iteration=i)
            ypredict = model.predict(X=xtest,baseName= convertDocName[base],iteration=i, isRuningZ=False)
            confMatrix = confusionMatrix(ytest, ypredict)
            print('Confusion Matrix:\n', confMatrix)
            plotConfusionMatrix(confMatrix,originalLabels,'kMeansQuant',i,convertDocName[base])
            accuracy = np.trace(confMatrix) / np.sum(confMatrix)
            print('ACC:', accuracy)
            arquivo.write("ACC: {}\n".format(accuracy))
            arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
            accuracyList.append(accuracy)
            plotDecisionSurface(xtrain, ytrain,'kMeansQuant',i,convertDocName[base])
        print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
        arquivo.write(
            '\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))


def BayesianRejectionRuns(base):
    convertRun = {
        0: openIrisDatasetRejectRun(),
        1: openColumnDatasetRejectRun(),
        2: openArtificialDatasetRejectRun()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial'
    }

    Wr_values = [0.04, 0.12, 0.24, 0.36, 0.48]
    out = convertRun[base]
    X = out[0]
    y = out[1]
    originalLabels = out[2]

    all_accuracy_lists = []
    all_rejection_rate_lists = []



    for Wr in Wr_values:
        accuracyList = []
        rejectionRateList = []
        fileName = f"DadosRuns/BayesianRejection_{convertDocName[base]}_Wr{Wr}.txt"
        with open(fileName, 'w') as arquivo:
            arquivo.write(f"Execução Iterações Bayesian Rejection {convertDocName[base]} Wr={Wr}.\n\n")
            for i in range(20):
                print(f'\nIteração {i}\n')
                X_train, y_train, X_test, y_test = datasetSplitTrainTest(X, y, 80)
                plotDataPoints(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test),
                               convertDocName[base], i, 'Train-Test', originalLabels)
                model = BayesianGaussianRejectionQuant(Wr=Wr)
                model.fit(X_train, y_train)
                accuracy, rejection_rate, y_pred = model.evaluate(X_test, y_test)

                accuracyList.append(accuracy)
                rejectionRateList.append(rejection_rate)
                arquivo.write(f"Iteração {i}: Acurácia = {accuracy:.4f}, Taxa de Rejeição = {rejection_rate:.4f}\n")
                confMatrix = confusionMatrix(y_test, y_pred)
                print('Confusion Matrix:\n', confMatrix)
                plotConfusionMatrix(confMatrix, originalLabels, f'BayesianGaussianRejection_Wr_{Wr}', i, convertDocName[base])
                accuracy = np.trace(confMatrix) / np.sum(confMatrix)
                print('ACC:', accuracy)
                arquivo.write("ACC: {}\n".format(accuracy))
                arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
                plotDecisionSurface(X_train, y_train, f'BayesianGaussianRejection_Wr_{Wr}', i, convertDocName[base], originalLabels)



            all_accuracy_lists.append(accuracyList)
            all_rejection_rate_lists.append(rejectionRateList)

            avg_accuracy = np.mean(accuracyList)
            avg_rejection_rate = np.mean(rejectionRateList)
            std_accuracy = np.std(accuracyList)
            std_rejection_rate = np.std(rejectionRateList)

            arquivo.write(f'\nAcurácia média das 20 iterações: {avg_accuracy:.2f} ± {std_accuracy:.2f}\n')
            print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(avg_accuracy, std_accuracy))
            arquivo.write(f'Taxa de Rejeição média das 20 iterações: {avg_rejection_rate:.2f} ± {std_rejection_rate:.2f}\n')

    plotAccuracyRejectionCurvePerWr(Wr_values, all_accuracy_lists, all_rejection_rate_lists, convertDocName[base])


def BayesianGaussianMixtureRuns(base):
    # Same dataset preparation as other runs
    convertRun = {
        0: openIrisDataset(),
        1: openColumnDataset(),
        2: openArtificialDataset(),
        3: openBreastDataset(),
        4: openDermatologyDataset()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial',
        3: 'Breast',
        4: 'Dermatology'
    }

    out = convertRun[base]
    x = out[0]
    y = out[1]
    originalLabels = out[2]
    accuracyList = []

    fileName = f"DadosRuns/BayesianGaussianMixtureRuns_{convertDocName[base]}.txt"
    with open(fileName, 'w') as arquivo:
        arquivo.write(f"Execução Iterações Bayesian Gaussian Mixture {convertDocName[base]}.\n\n")
        for i in range(20):
            print(f'\nIteração {i}\n')
            xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x, y, 80)
            plotDataPoints(np.array(xtrain), np.array(ytrain), np.array(xtest), np.array(ytest),
                           convertDocName[base], i, 'Train-Test', originalLabels)
            model = BayesianGaussianMixtureClassifier(n_components=3)
            model.fit(xtrain, ytrain, convertDocName[base], True, i)
            ypredict = model.predict(xtest, convertDocName[base], i, False)
            confMatrix = confusionMatrix(ytest, ypredict)
            print('Confusion Matrix:\n', confMatrix)
            plotConfusionMatrix(confMatrix, originalLabels, 'BayesianGaussianMixture', i, convertDocName[base])
            accuracy = np.trace(confMatrix) / np.sum(confMatrix)
            print('ACC:', accuracy)
            arquivo.write(f"ACC: {accuracy}\n")
            arquivo.write(f"Confusion Matrix: \n {confMatrix} \n\n")
            accuracyList.append(accuracy)
            plotDecisionSurface(xtrain, ytrain, 'BayesianGaussianMixture', i, convertDocName[base],originalLabels)
        print(f'\nAcurácia média das 20 iterações: {np.mean(accuracyList):.2f} ± {np.std(accuracyList):.2f}')
        arquivo.write(f'\nAcurácia média das 20 iterações: {np.mean(accuracyList):.2f} ± {np.std(accuracyList):.2f}')


def BayesianParzenRuns(base):
    from openDatasets import openIrisDataset, openColumnDataset, openArtificialDataset, openBreastDataset, openDermatologyDataset, datasetSplitTrainTest
    convertRun = {
        0: openIrisDataset(),
        1: openColumnDataset(),
        2: openArtificialDataset(),
        3: openBreastDataset(),
        4: openDermatologyDataset()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial',
        3: 'Breast',
        4: 'Dermatology'
    }

    out = convertRun[base]
    X = out[0]
    y = out[1]
    originalLabels = out[2]
    accuracyList = []

    fileName = "DadosRuns/BayesianParzenRuns_{}.txt".format(convertDocName[base])
    with open(fileName, 'w') as arquivo:
        arquivo.write("Execução Iterações Parzen Window {}.\n\n".format(convertDocName[base]))
        for i in range(20):
            print('\nIteração {}\n'.format(i))
            X_train, y_train, X_test, y_test = datasetSplitTrainTest(X, y, 80)
            plotDataPoints(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test),
                           convertDocName[base], i, 'Train-Test', originalLabels, FolderName='ParzenWindow')
            model = BayesianParzenClassifier(bandwidth=1.0,epsilon=1e-5)
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            confMatrix = confusionMatrix(y_test, y_predict)
            print('Confusion Matrix:\n', confMatrix)
            plotConfusionMatrix(confMatrix, originalLabels, 'ParzenWindow', i, convertDocName[base])
            accuracy = np.trace(confMatrix) / np.sum(confMatrix)
            print('ACC:', accuracy)
            arquivo.write(f"ACC: {accuracy}\n")
            arquivo.write(f"Confusion Matrix: \n {confMatrix} \n\n")
            accuracyList.append(accuracy)
            plotDecisionSurface(X_train, y_train, 'ParzenWindow', i, convertDocName[base],originalLabels)
        print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
        arquivo.write('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
