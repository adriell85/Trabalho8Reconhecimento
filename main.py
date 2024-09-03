import matplotlib
matplotlib.use('TkAgg')
from runs import KNNRuns,DMCRuns,BayesianGaussianDiscriminantRuns,KmeansQuantRuns, BayesianRejectionRuns,BayesianGaussianMixtureRuns,BayesianParzenRuns


def main():
    # BayesianParzenRuns(0)
    # BayesianParzenRuns(1)
    # BayesianParzenRuns(2)
    BayesianParzenRuns(3)
    # BayesianParzenRuns(4)

if __name__ == "__main__":
    main()
