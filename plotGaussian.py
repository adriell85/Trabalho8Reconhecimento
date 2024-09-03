import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

def plotGaussianDistribution(means, covariances, classes, featureIndices=(0, 1), gridRange=(-3, 3), resolution=0.1):

    f1, f2 = featureIndices
    x, y = np.mgrid[gridRange[0]:gridRange[1]:resolution, gridRange[0]:gridRange[1]:resolution]
    pos = np.dstack((x, y))

    fig, ax = plt.subplots()

    for i, c in enumerate(classes):
        mean = means[i][[f1, f2]]
        covariance = covariances[i][[f1, f2], [f1, f2]]
        rv = multivariate_normal(mean, covariance)
        ax.contourf(x, y, rv.pdf(pos), levels=100, cmap='Blues', alpha=0.5)
        ax.set_title(f'Multivariate Gaussian Distribution - Features {f1} and {f2}')
        ax.set_xlabel(f'Feature {f1}')
        ax.set_ylabel(f'Feature {f2}')

    plt.show()



def plotGaussianDistribution3d(modeName,baseName,iteration,means, covariances, classes, featureIndices=(0, 1), gridRange=(-0.3, 0.3), resolution=0.1):
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
    if (baseName == 'Iris'):
        atributesCombination = atributesCombinationIris
    elif (baseName == 'Artificial'):
        atributesCombination = atributesCombinationArtificial
    else:
        atributesCombination = atributesCombinationFree
    for ind in atributesCombination:
        f1, f2 = ind
        x, y = np.mgrid[gridRange[0]:gridRange[1]:resolution, gridRange[0]:gridRange[1]:resolution]
        pos = np.dstack((x, y))

        fig2 = plt.figure()


        for i, c in enumerate(classes):
            ax = fig2.add_subplot(111, projection='3d')
            mean = means[i][[f1, f2]]
            covariance = covariances[i][[f1, f2], [f1, f2]]

            rv = multivariate_normal(mean=mean, cov=covariance, allow_singular=True)
            z = rv.pdf(pos)

            ax.plot_surface(x, y, z, cmap='cividis', edgecolor='none', alpha=0.5)
            ax.set_title(f'Multivariate Gaussian Distribution - Features {f1} and {f2} base {baseName}')
            ax.set_xlabel(f'Feature {f1}')
            ax.set_ylabel(f'Feature {f2}')
            ax.set_zlabel('Probability')
            plt.savefig('Resultados_Bayes_{}/{}/Gaussiana_{}_Base_{}_features_{}_classe_{}_iteracao_{}.png'.format(modeName,baseName,modeName,baseName,ind,i,iteration))

        # plt.show()

def dispersionDataByClass(data, datasetName,iteration,classIndex,modeName):
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
    color_map = {
        0: "red",
        1: "blue",
        2: "green",
        3: "yellow",
        4: "purple",
        5: "orange",
        6: "black"
    }
    if datasetName == 'Iris':
        atributesCombination = atributesCombinationIris
    elif datasetName == 'Artificial':
        atributesCombination = atributesCombinationArtificial
    else:
        atributesCombination = atributesCombinationFree

    num_plots = len(atributesCombination)
    cols = 3
    rows = (num_plots + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.ravel()



    for i, (idx1, idx2) in enumerate(atributesCombination):
        x = [row[idx1] for row in data]
        y = [row[idx2] for row in data]
        axs[i].scatter(x, y,color=color_map[classIndex])
        axs[i].set_xlabel(f'Atributo {idx1}')
        axs[i].set_ylabel(f'Atributo {idx2}')
        axs[i].set_title(f'Atributo {idx1} e Atributo {idx2}')

    for j in range(i + 1, rows * cols):
        fig.delaxes(axs[j])

    fig.suptitle('Base {}, iteração {}'.format(datasetName, iteration))

    plt.tight_layout()

    plt.savefig('Resultados_Bayes_{}/{}/Grafico_dispersao_Dados_Treino_Base_{}_iteracao_{}_classe_{}'.format(modeName,modeName,datasetName,datasetName, iteration,classIndex))


def dispersionDataBlindClass(data, datasetName,iteration,isTrainingData,modeName):
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


    num_plots = len(atributesCombination)
    cols = 3
    rows = (num_plots + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.ravel()  # Transforma o array multidimensional em 1D para facilitar o acesso


    for i, (idx1, idx2) in enumerate(atributesCombination):
        x = [row[idx1] for row in data]
        y = [row[idx2] for row in data]
        axs[i].scatter(x, y)
        axs[i].set_xlabel(f'Atributo {idx1}')
        axs[i].set_ylabel(f'Atributo {idx2}')
        axs[i].set_title(f'Atributo {idx1} e Atributo {idx2}')

    # Esconder subplots extras que não estão em uso
    for j in range(i + 1, rows * cols):
        fig.delaxes(axs[j])

    fig.suptitle('Base {}, iteração {}'.format(datasetName,iteration ))

    plt.tight_layout()
    if(isTrainingData):
        plt.savefig('Resultados_{}/{}/Grafico_dispersao_Dados_Treino_Base_{}_{}_iteracao_{}'.format(modeName,datasetName,datasetName,modeName,iteration))
    else:
        plt.savefig(
            'Resultados_{}/{}/Grafico_dispersao_Dados_Teste_Base_{}_{}_iteracao_{}'.format(modeName,datasetName, datasetName,modeName,iteration))
    # plt.show()

def plotDadosColridos(xtrain, ytrain, datasetName, iteration, isTrainingData, modeName):
    # Attribute combinations for different datasets
    atributesCombination = {
        'Artificial': [[0, 1]],
        'Iris': [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        'Free': [[0, 1], [0, 4], [0, 5], [2, 3], [3, 4], [4, 5]]
    }

    # Get attribute combinations based on dataset
    attribute_combinations = atributesCombination.get(datasetName, atributesCombination['Free'])

    # Number of plots
    num_plots = len(attribute_combinations)
    cols = 3
    rows = (num_plots + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.ravel()  # Flatten the axes array for easier indexing

    # Generate scatter plots for each attribute combination
    for i, (idx1, idx2) in enumerate(attribute_combinations):
        for class_value in set(ytrain):  # Iterate through each class
            # Select data for the current class
            class_indices = [index for index, value in enumerate(ytrain) if value == class_value]
            x = [xtrain[index][idx1] for index in class_indices]
            y = [xtrain[index][idx2] for index in class_indices]

            axs[i].scatter(x, y, label=f'Class {class_value}')

        axs[i].set_xlabel(f'Attribute {idx1}')
        axs[i].set_ylabel(f'Attribute {idx2}')
        axs[i].set_title(f'Attribute {idx1} vs Attribute {idx2}')
        axs[i].legend()

    # Hide unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axs[j])

    fig.suptitle(f'Dataset {datasetName}, Iteration {iteration}')
    plt.tight_layout()
    plot_path = f'Resultados_{modeName}/{datasetName}/Grafico_dispersao_colorido_Data_{"Treino" if isTrainingData else "Teste"}_Dataset_{datasetName}_{modeName}_Iteracao{iteration}.png'
    plt.savefig(plot_path)
    # plt.show()

def plotCovarianceMatrix(cov_matrix, baseName,iteration, modeName):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Covariance Matrix')
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.savefig(
        'Resultados_{}/{}/Matriz_de_Covariancia_Base_{}_{}_iteracao_{}'.format(modeName, baseName,
                                                                                       baseName, modeName,
                                                                                       iteration))
    # plt.show()
