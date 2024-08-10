import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.twodigit.train_utils import batchify_data, run_epoch, train_model, Flatten
import src.twodigit.utils_multiMNIST as U
path_to_data_dir = 'datasets/'
use_mini_dataset = True

batch_size = 128
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        channels1 = 32
        channels2 = 64

        self.first1 = nn.Conv2d(1, channels1, (3, 3))
        self.first2 = nn.LeakyReLU()
        self.first3 = nn.MaxPool2d((2, 2))
        self.first4 = nn.Conv2d(channels1, channels2, (3, 3))
        self.first5 = nn.LeakyReLU()
        self.first6 = nn.MaxPool2d((2, 2))
        self.first7 = nn.Flatten()
        self.first8 = nn.Linear(2880, 128)
        self.first9 = nn.Dropout(p=0.75)
        self.first12 = nn.Linear(128, num_classes)

        self.second1 = nn.Conv2d(1, channels1, (3, 3))
        self.second2 = nn.LeakyReLU()
        self.second3 = nn.MaxPool2d((2, 2))
        self.second4 = nn.Conv2d(channels1, channels2, (3, 3))
        self.second5 = nn.LeakyReLU()
        self.second6 = nn.MaxPool2d((2, 2))
        self.second7 = nn.Flatten()
        self.second8 = nn.Linear(2880, 128)
        self.second9 = nn.Dropout(p=0.75)
        self.second12 = nn.Linear(128, num_classes)

    def forward(self, x):

        out_first_digit = self.first1(x)
        out_first_digit = self.first2(out_first_digit)
        out_first_digit = self.first3(out_first_digit)
        out_first_digit = self.first4(out_first_digit)
        out_first_digit = self.first5(out_first_digit)
        out_first_digit = self.first6(out_first_digit)
        out_first_digit = self.first7(out_first_digit)
        out_first_digit = self.first8(out_first_digit)
        out_first_digit = self.first9(out_first_digit)
        out_first_digit = self.first12(out_first_digit)

        out_second_digit = self.second1(x)
        out_second_digit = self.second2(out_second_digit)
        out_second_digit = self.second3(out_second_digit)
        out_second_digit = self.second4(out_second_digit)
        out_second_digit = self.second5(out_second_digit)
        out_second_digit = self.second6(out_second_digit)
        out_second_digit = self.second7(out_second_digit)
        out_second_digit = self.second8(out_second_digit)
        out_second_digit = self.second9(out_second_digit)
        out_second_digit = self.second12(out_second_digit)

        return out_first_digit, out_second_digit

def run_cnn():
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
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

    if torch.cuda.is_available():
        model.to("cuda")

    # Train
    lr = 0.08
    train_model(train_batches, dev_batches, model, lr)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

