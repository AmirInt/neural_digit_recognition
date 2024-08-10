import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.twodigit.train_utils import batchify_data, run_epoch, train_model, Flatten
import src.twodigit.utils_multiMNIST as U
path_to_data_dir = "datasets/"
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions

class MLP(nn.Module):

    def __init__(self, input_dimension):
        super(MLP, self).__init__()
        self.flatten = Flatten()
        self.hidden1 = nn.Linear(input_dimension, 64)
        self.hidden2 = nn.Linear(input_dimension, 64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.output1 = nn.Linear(64, 10)
        self.output2 = nn.Linear(64, 10)

    def forward(self, x):
        xf = self.flatten(x)
        
        out_first_digit = self.hidden1(xf)
        out_first_digit = self.relu1(out_first_digit)
        out_first_digit = self.output1(out_first_digit)
        
        out_second_digit = self.hidden2(xf)
        out_second_digit = self.relu2(out_second_digit)
        out_second_digit = self.output2(out_second_digit)
        
        return out_first_digit, out_second_digit

def run_mlp():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = MLP(input_dimension) # TODO add proper layers to MLP class above

    if torch.cuda.is_available():
        model.to("cuda")

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))
