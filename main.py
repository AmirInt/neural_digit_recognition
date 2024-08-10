import sys
import torch
from src.nn.neural_nets import *
import src.mnist.nnet_fc as nnet_fc
import src.mnist.nnet_cnn as nnet_cnn
import src.twodigit.mlp as mlp
import src.twodigit.conv as conv


def display_usage():
    print("Usage: python main.py [option]")
    print("Options:")
    print("nn: Run the initial simple neural network")
    print("nnet_fc: Run the FCNN on MNIST")
    print("nnet_cnn: Run the CNN on MNIST")
    print("twodigit_mlp: Run the FCNN on two-digit MNIST")
    print("twodigit_cnn: Run the CNN on two-digit MNIST")


def main():
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    try:
        if sys.argv[1] == "nn":        
            x = NeuralNetwork()
            x.train_neural_network()
            x.test_neural_network()
        elif sys.argv[1] == "nnet_fc":
            nnet_fc.run_nnet_fc()
        elif sys.argv[1] == "nnet_cnn":
            nnet_cnn.run_nnet_cnn()
        elif sys.argv[1] == "twodigit_mlp":
            mlp.run_mlp()
        elif sys.argv[1] == "twodigit_cnn":
            conv.run_cnn()
        else:
            display_usage()
    except IndexError:
        display_usage()

if __name__ == "__main__":
    main()
