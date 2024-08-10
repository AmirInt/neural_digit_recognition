import sys
import torch
from src.nn.neural_nets import *
import src.mnist.nnet_fc as nnet_fc
import src.mnist.nnet_cnn as nnet_cnn


def main():
    if sys.argv[1] == "nn":        
        x = NeuralNetwork()
        x.train_neural_network()
        x.test_neural_network()
    elif sys.argv[1] == "nnet_fc":
        # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
        np.random.seed(12321)  # for reproducibility
        torch.manual_seed(12321)  # for reproducibility
        nnet_fc.run_nnet_fc()
    elif sys.argv[1] == "nnet_cnn":
        # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
        np.random.seed(12321)  # for reproducibility
        torch.manual_seed(12321)
        nnet_cnn.run_nnet_cnn()

if __name__ == "__main__":
    main()
