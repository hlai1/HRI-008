""" Program to split the data and save it. """

import numpy as np
import pandas as pd


# Splits all experiment data 80/20 (for model training)
def split_data_80_20(experiment_data):
    msk = np.random.rand(len(experiment_data)) < 0.8
    train = experiment_data[msk]
    test = experiment_data[~msk]

    return train, test


# Splits all experiment data 50/50 (for model evaluation)
def split_data_50_50(experiment_data):
    msk = np.random.rand(len(experiment_data)) < 0.5
    train = experiment_data[msk]
    test = experiment_data[~msk]

    return train, test

'''
# Split experiment 1 data 80/20
experiment_1_data = pd.read_csv('data/experiment_data/ex1_data.txt', header=None)
ex1_train_data, ex1_test_data = split_data_80_20(experiment_1_data)
ex1_train_data.to_csv('data/training_data/ex1_training_data.txt', index=False, header=None)
ex1_test_data.to_csv('data/testing_data/ex1_testing_data.txt', index=False, header=None)

# Split experiment 2 data 80/20
experiment_2_data = pd.read_csv('data/experiment_data/ex2_data.txt', header=None)
ex2_train_data, ex2_test_data = split_data_80_20(experiment_2_data)
ex2_train_data.to_csv('data/training_data/ex2_training_data.txt', index=False, header=None)
ex2_test_data.to_csv('data/testing_data/ex2_testing_data.txt', index=False, header=None)

# Split experiment 3 data 80/20
experiment_3_data = pd.read_csv('data/experiment_data/ex3_data.txt', header=None)
ex3_train_data, ex3_test_data = split_data_80_20(experiment_3_data)
ex3_train_data.to_csv('data/training_data/ex3_training_data.txt', index=False, header=None)
ex3_test_data.to_csv('data/testing_data/ex3_testing_data.txt', index=False, header=None)

# Split experiment 4 data 80/20
experiment_4_data = pd.read_csv('data/experiment_data/ex4_data.txt', header=None)
ex4_train_data, ex4_test_data = split_data_80_20(experiment_4_data)
ex4_train_data.to_csv('data/training_data/ex4_training_data.txt', index=False, header=None)
ex4_test_data.to_csv('data/testing_data/ex4_testing_data.txt', index=False, header=None)

# Split experiment 5 data 80/20
experiment_5_data = pd.read_csv('data/experiment_data/ex5_data.txt', header=None)
ex5_train_data, ex5_test_data = split_data_80_20(experiment_5_data)
ex5_train_data.to_csv('data/training_data/ex5_training_data.txt', index=False, header=None)
ex5_test_data.to_csv('data/testing_data/ex5_testing_data.txt', index=False, header=None)

# Split experiment 6 data 80/20
experiment_6_data = pd.read_csv('data/experiment_data/ex6_data.txt', header=None)
ex6_train_data, ex6_test_data = split_data_80_20(experiment_6_data)
ex6_train_data.to_csv('data/training_data/ex6_training_data.txt', index=False, header=None)
ex6_test_data.to_csv('data/testing_data/ex6_testing_data.txt', index=False, header=None)
'''

all_experiment_data = pd.read_csv('data/experiment_data/all_experiment_data.txt', header=None)

# Split the training/testing data 50/50 5 times (for 5 x 2 CV)
train_data_0, test_data_0 = split_data_50_50(all_experiment_data)
train_data_0.to_csv('data/training_data/cv1_training_data.txt', index=False, header=None)
test_data_0.to_csv('data/testing_data/cv1_testing_data.txt', index=False, header=None)

train_data_1, test_data_1 = split_data_50_50(all_experiment_data)
train_data_1.to_csv('data/training_data/cv2_training_data.txt', index=False, header=None)
test_data_1.to_csv('data/testing_data/cv2_testing_data.txt', index=False, header=None)

train_data_2, test_data_2 = split_data_50_50(all_experiment_data)
train_data_2.to_csv('data/training_data/cv3_training_data.txt', index=False, header=None)
test_data_2.to_csv('data/testing_data/cv3_testing_data.txt', index=False, header=None)

train_data_3, test_data_3 = split_data_50_50(all_experiment_data)
train_data_3.to_csv('data/training_data/cv4_training_data.txt', index=False, header=None)
test_data_3.to_csv('data/testing_data/cv4_testing_data.txt', index=False, header=None)

train_data_4, test_data_4 = split_data_50_50(all_experiment_data)
train_data_4.to_csv('data/training_data/cv5_training_data.txt', index=False, header=None)
test_data_4.to_csv('data/testing_data/cv5_testing_data.txt', index=False, header=None)
