import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('TkAgg')
from NaiveBayes import NaiveBayesClassifier
from KNN import KNN
from DMC import DMC
import matplotlib.pyplot as plt
import os


def confusionMatrix(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    num_classes = max(max(y_true), max(y_pred)) + 1
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        conf_matrix[true][pred] += 1

    return conf_matrix

def plotConfusionMatrix(conf_matrix, class_names,classifierName,i,datasetName):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('**True Label**')
    plt.xlabel('**Predicted Label**')
    plt.title('Confusion Matrix')
    plt.savefig('Resultados_{}/{}/Matriz_de_Confusao_base_{}_Iteracao_{}.png'.format(classifierName,datasetName,datasetName,i))


def plotDecisionSurface(xtrain, ytrain, classifierName, i, datasetName, class_labels):
    atributesCombinationArtificial = [
        [0, 1]
    ]
    atributesCombinationIris = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3]
    ]
    atributesCombinationFree = [
        [0, 1],
        [0, 4],
        [0, 5],
        [2, 3],
        [3, 4],
        [4, 5]
    ]

    if datasetName == 'Iris':
        atributesCombination = atributesCombinationIris
    elif datasetName == 'Artificial':
        atributesCombination = atributesCombinationArtificial
    else:
        atributesCombination = atributesCombinationFree

    for z in atributesCombination:
        xtrainSelected = np.array(xtrain)[:, z]
        x_min = xtrainSelected[:, 0].min() - 1
        x_max = xtrainSelected[:, 0].max() + 1
        y_min = xtrainSelected[:, 1].min() - 1
        y_max = xtrainSelected[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        matrix = np.c_[xx.ravel(), yy.ravel()]

        if classifierName == 'KNN':
            Z = KNN(xtrainSelected, ytrain, matrix, k=3)
        elif classifierName == 'DMC':
            Z = DMC(xtrainSelected, ytrain, matrix)
        else:
            model = NaiveBayesClassifier()
            model.fit(xtrainSelected, ytrain, datasetName, False, i)
            Z = model.predict(matrix, datasetName, i, True)

        Z = np.array(Z).reshape(xx.shape)
        fig, ax = plt.subplots()
        colors = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=colors)

        scatter = plt.scatter(xtrainSelected[:, 0], xtrainSelected[:, 1], c=ytrain, s=20, edgecolor='k', cmap=colors)

        handles, _ = scatter.legend_elements()
        plt.legend(handles, class_labels, title="Classes")

        plt.title('Superfície de Decisão do {} base {}'.format(classifierName, datasetName))
        plt.xlabel('Atributo 1')
        plt.ylabel('Atributo 2')


        output_dir = f'Resultados_{classifierName}/{datasetName}'
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f'{output_dir}/Superficie_de_decisao_base_{datasetName}_Atributos_{z}_Iteracao_{i}.png')
        plt.close(fig)


def plotAccuracyRejectionCurvePerWr(Wr_values, all_accuracy_lists, all_rejection_rate_lists, baseName):
    plt.figure(figsize=(10, 6))
    avg_accuracies = []
    avg_rejection_rates = []

    # Itera sobre cada Wr para calcular médias e plotar pontos
    for i, Wr in enumerate(Wr_values):
        accuracies = all_accuracy_lists[i]
        rejection_rates = all_rejection_rate_lists[i]
        avg_accuracy = np.mean(accuracies)
        avg_rejection_rate = np.mean(rejection_rates)
        avg_accuracies.append(avg_accuracy)
        avg_rejection_rates.append(avg_rejection_rate)

    # Plota a curva conectando os pontos
    plt.plot(avg_rejection_rates, avg_accuracies, marker='o', linestyle='-', label='Curva Acurácia-Rejeição')

    # Adiciona anotações para cada Wr
    for i, Wr in enumerate(Wr_values):
        plt.annotate(f'Wr={Wr}', (avg_rejection_rates[i], avg_accuracies[i]))

    plt.title(f'Curva Acurácia-Rejeição (AR) por Wr - {baseName}')
    plt.xlabel('Taxa de Rejeição')
    plt.ylabel('Acurácia')
    plt.grid(True)
    plt.legend()

    # Cria o diretório e salva o gráfico
    os.makedirs(f"Resultados_BayesianGaussianRejectionQuant/{baseName}", exist_ok=True)
    plt.savefig(f"Resultados_BayesianGaussianRejectionQuant/{baseName}/Curva_AR_por_Wr_{baseName}.png")
    # plt.show()

def plotDataPoints(X_train, y_train, X_test, y_test, baseName, iteration, phase, originalLabels,FolderName):
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)
    plt.figure(figsize=(8, 6))

    # Plotando os dados de treino com legenda por classe
    classes = np.unique(y_train)
    for i, class_label in enumerate(classes):
        if np.any(y_train == class_label):
            plt.scatter(X_train[y_train == class_label, 0], X_train[y_train == class_label, 1],
                        label=f'Train - {originalLabels[class_label]}', s=50, edgecolor='k', cmap='viridis')

    classes_test = np.unique(y_test)
    for i, class_label in enumerate(classes_test):
        if np.any(y_test == class_label):
            plt.scatter(X_test[y_test == class_label, 0], X_test[y_test == class_label, 1],
                        label=f'Test - {originalLabels[class_label]}', s=100, edgecolor='k', marker='x',
                        cmap='coolwarm')

    plt.title(f'Distribuição de Dados ({phase}) - Iteration {iteration}')
    plt.xlabel('Atributo 1')
    plt.ylabel('Atributo 2')
    plt.legend()
    plt.grid(True)

    os.makedirs(f"Resultados_{baseName}", exist_ok=True)
    plt.savefig(
        f"Resultados_{FolderName}/{baseName}/Distribuicao_Dados_{baseName}_{phase}_Iteration_{iteration}.png")


def plotClusters(self, X, labels, baseName, iteration, phase):
        # X = np.array(X)
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

                Z = self.predict(grid_points_expanded, isRuningZ=True)
                Z = Z.reshape(xx.shape)

                plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

                plt.title(f'Bayesian Parzen Window ({phase}) - Iteration {iteration} (Features {feature_1 + 1} vs {feature_2 + 1})')
                plt.xlabel(f'Feature {feature_1 + 1}')
                plt.ylabel(f'Feature {feature_2 + 1}')
                os.makedirs(f"Resultados_ParzenWindow/{baseName}", exist_ok=True)
                plt.savefig(f"Resultados_ParzenWindow/{baseName}/Bayesian_{baseName}_{phase}_Iteration_{iteration}_Features_{feature_1 + 1}_vs_{feature_2 + 1}.png")
                plt.close()
