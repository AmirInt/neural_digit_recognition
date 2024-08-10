import sys
from src.nn.neural_nets import *


def main():
    if sys.argv[1] == "nn":        
        x = NeuralNetwork()
        x.train_neural_network()
        x.test_neural_network()


if __name__ == "__main__":
    main()
