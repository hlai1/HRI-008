"""
This is a program for calculating the CPTs that will be used for model training and evaluation.

Conditional Probability Distribution Table:

P(D|PFH,IA,TP)  P(B|PFH,IA,TP)  P(P|PFH,IA,TP)  = 1
P(D|PFH,IA)     P(B|PFH,IA)     P(P|PFH,IA)     = 1
P(D|PFH,TP)     P(B|PFH,TP)     P(P|PFH,TP)     = 1
P(D|IA,TP)      P(B|IA,TP)      P(P|IA,TP)      = 1
P(D|PFH)        P(B|PFH)        P(P|PFH)        = 1
P(D|TP)         P(B|TP)         P(P|TP)         = 1
P(D|IA)         P(B|IA)         P(P|IA)         = 1
P(D|None)       P(B|None)       P(P|None)       = 1

"""

import numpy as np
import pandas as pd
# from sklearn.model_selection import KFold
# import collections
# from scipy import stats
# from sklearn import metrics


def open_data(filename):
    with open(filename, 'r'):
        lines = [p.split() for p in [line.rstrip('\n') for line in open(filename)]]

    return lines


def calculate_model_outputs(data):

    # Create a new dataframe for the probability distribution
    experiment_probs = pd.DataFrame(np.zeros(shape=(8, 4)), columns=['contextual_factor', 'D', 'B', 'P'])
    experiment_probs['contextual_factor'] = ['PFH,IA,TP', 'PFH,IA', 'PFH,TP', 'IA,TP', 'PFH', 'IA', 'TP', 'None']
    experiment_probs = experiment_probs.set_index('contextual_factor', drop=True)

    # Initialize the total number of utterances per contextual factor
    total_pfh_ia_tp, total_pfh_ia, total_pfh_tp, total_ia_tp, total_pfh, total_ia, total_tp, total_none = 0, 0, 0, 0, 0, 0, 0, 0

    # Initialize the total number of utterances per norm: D
    d_pfh_ia_tp, d_pfh_ia, d_pfh_tp, d_ia_tp, d_pfh, d_ia, d_tp, d_none = 0, 0, 0, 0, 0, 0, 0, 0

    # Initialize the total number of utterances per norm: B
    b_pfh_ia_tp, b_pfh_ia, b_pfh_tp, b_ia_tp, b_pfh, b_ia, b_tp, b_none = 0, 0, 0, 0, 0, 0, 0, 0

    # Initialize the total number of utterances per norm: P
    p_pfh_ia_tp, p_pfh_ia, p_pfh_tp, p_ia_tp, p_pfh, p_ia, p_tp, p_none = 0, 0, 0, 0, 0, 0, 0, 0

    # Populate the dataframe with the annotations
    for line in data:

        ia = line[2][0]  # IA: Y or N
        pfh = line[1][0]  # PFH: Y or N
        tp = line[3][0]  # TP: Y or N
        direct = line[4][0]  # D: Y or N
        brief = line[5][0]  # B: Y or N
        polite = line[6]  # P: Y or N

        if pfh == 'Y' and ia == 'Y' and tp == 'Y' and direct == 'Y':
            d_pfh_ia_tp += 1
            total_pfh_ia_tp += 1
        elif pfh == 'Y' and ia == 'Y' and tp == 'N' and direct == 'Y':
            d_pfh_ia += 1
            total_pfh_ia += 1
        elif pfh == 'Y' and ia == 'N' and tp == 'Y' and direct == 'Y':
            d_pfh_tp += 1
            total_pfh_tp += 1
        elif pfh == 'N' and ia == 'Y' and tp == 'Y' and direct == 'Y':
            d_ia_tp += 1
            total_ia_tp += 1
        elif pfh == 'Y' and ia == 'N' and tp == 'N' and direct == 'Y':
            d_pfh += 1
            total_pfh += 1
        elif pfh == 'N' and ia == 'Y' and tp == 'N' and direct == 'Y':
            d_ia += 1
            total_ia += 1
        elif pfh == 'N' and ia == 'N' and tp == 'Y' and direct == 'Y':
            d_tp += 1
            total_tp += 1
        elif pfh == 'N' and ia == 'N' and tp == 'N' and direct == 'Y':
            d_none += 1
            total_none += 1

        elif pfh == 'Y' and ia == 'Y' and tp == 'Y' and brief == 'Y':
            b_pfh_ia_tp += 1
            total_pfh_ia_tp += 1
        elif pfh == 'Y' and ia == 'Y' and tp == 'N' and brief == 'Y':
            b_pfh_ia += 1
            total_pfh_ia += 1
        elif pfh == 'Y' and ia == 'N' and tp == 'Y' and brief == 'Y':
            b_pfh_tp += 1
            total_pfh_tp += 1
        elif pfh == 'N' and ia == 'Y' and tp == 'Y' and brief == 'Y':
            b_ia_tp += 1
            total_ia_tp += 1
        elif pfh == 'Y' and ia == 'N' and tp == 'N' and brief == 'Y':
            b_pfh += 1
            total_pfh += 1
        elif pfh == 'N' and ia == 'Y' and tp == 'N' and brief == 'Y':
            b_ia += 1
            total_ia += 1
        elif pfh == 'N' and ia == 'N' and tp == 'Y' and brief == 'Y':
            b_tp += 1
            total_tp += 1
        elif pfh == 'N' and ia == 'N' and tp == 'N' and brief == 'Y':
            b_none += 1
            total_none += 1

        elif pfh == 'Y' and ia == 'Y' and tp == 'Y' and polite == 'Y':
            p_pfh_ia_tp += 1
            total_pfh_ia_tp += 1
        elif pfh == 'Y' and ia == 'Y' and tp == 'N' and polite == 'Y':
            p_pfh_ia += 1
            total_pfh_ia += 1
        elif pfh == 'Y' and ia == 'N' and tp == 'Y' and polite == 'Y':
            p_pfh_tp += 1
            total_pfh_tp += 1
        elif pfh == 'N' and ia == 'Y' and tp == 'Y' and polite == 'Y':
            p_ia_tp += 1
            total_ia_tp += 1
        elif pfh == 'Y' and ia == 'N' and tp == 'N' and polite == 'Y':
            p_pfh += 1
            total_pfh += 1
        elif pfh == 'N' and ia == 'Y' and tp == 'N' and polite == 'Y':
            p_ia += 1
            total_ia += 1
        elif pfh == 'N' and ia == 'N' and tp == 'Y' and polite == 'Y':
            p_tp += 1
            total_tp += 1
        elif pfh == 'N' and ia == 'N' and tp == 'N' and polite == 'Y':
            p_none += 1
            total_none += 1

    if total_pfh_ia_tp > 0:
        experiment_probs['D']['PFH,IA,TP'] = d_pfh_ia_tp / total_pfh_ia_tp
        experiment_probs['B']['PFH,IA,TP'] = b_pfh_ia_tp / total_pfh_ia_tp
        experiment_probs['P']['PFH,IA,TP'] = p_pfh_ia_tp / total_pfh_ia_tp
        # print("PFH,IA,TP:", total_pfh_ia_tp)
    else:
        experiment_probs['D']['PFH,IA,TP'] = 0
        experiment_probs['B']['PFH,IA,TP'] = 0
        experiment_probs['P']['PFH,IA,TP'] = 0

    experiment_probs['D']['PFH,IA'] = d_pfh_ia / total_pfh_ia
    experiment_probs['B']['PFH,IA'] = b_pfh_ia / total_pfh_ia
    experiment_probs['P']['PFH,IA'] = p_pfh_ia / total_pfh_ia
    # print("PFH,IA:", total_pfh_ia)

    if total_pfh_tp > 0:
        experiment_probs['D']['PFH,TP'] = d_pfh_tp / total_pfh_tp
        experiment_probs['B']['PFH,TP'] = b_pfh_tp / total_pfh_tp
        experiment_probs['P']['PFH,TP'] = p_pfh_tp / total_pfh_tp
        # print("PFH,TP:", total_pfh_tp)

    else:
        experiment_probs['D']['PFH,TP'] = 0
        experiment_probs['B']['PFH,TP'] = 0
        experiment_probs['P']['PFH,TP'] = 0

    if total_ia_tp > 0:
        experiment_probs['D']['IA,TP'] = d_ia_tp / total_ia_tp
        experiment_probs['B']['IA,TP'] = b_ia_tp / total_ia_tp
        experiment_probs['P']['IA,TP'] = p_ia_tp / total_ia_tp
        # print("IA,TP:", total_ia_tp)

    else:
        experiment_probs['D']['IA,TP'] = 0
        experiment_probs['B']['IA,TP'] = 0
        experiment_probs['P']['IA,TP'] = 0

    experiment_probs['D']['PFH'] = d_pfh / total_pfh
    experiment_probs['B']['PFH'] = b_pfh / total_pfh
    experiment_probs['P']['PFH'] = p_pfh / total_pfh
    # print("PFH:", total_pfh)

    experiment_probs['D']['IA'] = d_ia / total_ia
    experiment_probs['B']['IA'] = b_ia / total_ia
    experiment_probs['P']['IA'] = p_ia / total_ia
    # print("IA:", total_ia)

    if total_tp > 0:
        experiment_probs['D']['TP'] = d_tp / total_tp
        experiment_probs['B']['TP'] = b_tp / total_tp
        experiment_probs['P']['TP'] = p_tp / total_tp
        # print("TP:", total_tp)

    else:
        experiment_probs['D']['TP'] = 0
        experiment_probs['B']['TP'] = 0
        experiment_probs['P']['TP'] = 0

    experiment_probs['D']['None'] = d_none / total_none
    experiment_probs['B']['None'] = b_none / total_none
    experiment_probs['P']['None'] = p_none / total_none
    # print("None:", total_none)

    return experiment_probs


def create_ranking_matrix(model_results):
    # Create Ranking Matrix to Compare Models (training vs testing):
    rankings = model_results

    for index, row in model_results.iterrows():
        if row['D'] >= row['B'] >= row['P']:
            rankings.loc[index, 'D'] = 1
            rankings.loc[index, 'B'] = 2
            rankings.loc[index, 'P'] = 3
        elif row['D'] >= row['P'] >= row['B']:
            rankings.loc[index, 'D'] = 1
            rankings.loc[index, 'B'] = 3
            rankings.loc[index, 'P'] = 2
        elif row['B'] >= row['D'] >= row['P']:
            rankings.loc[index, 'D'] = 2
            rankings.loc[index, 'B'] = 1
            rankings.loc[index, 'P'] = 3
        elif row['P'] >= row['B'] >= row['D']:
            rankings.loc[index, 'D'] = 3
            rankings.loc[index, 'B'] = 2
            rankings.loc[index, 'P'] = 1
        elif row['B'] >= row['P'] >= row['D']:
            rankings.loc[index, 'D'] = 3
            rankings.loc[index, 'B'] = 1
            rankings.loc[index, 'P'] = 2
        elif row['P'] >= row['D'] >= row['B']:
            rankings.loc[index, 'D'] = 2
            rankings.loc[index, 'B'] = 3
            rankings.loc[index, 'P'] = 1

    return rankings


def compare_rankings(training_ranks, testing_ranks):

    for col in training_ranks:
        for row in training_ranks[col].index:
            if training_ranks.loc[row, col] != testing_ranks.loc[row, col]:
                print("Index :", row)
                print("Training Rankings: ", col, training_ranks.loc[row, col])
                print("Testing Rankings: ", col, testing_ranks.loc[row, col])


''' Model Training '''

# Experiment 1
ex1_train = open_data('data/training_data/ex1_training_data.txt')
ex1_test = open_data('data/testing_data/ex1_testing_data.txt')

# Experiment 2
ex2_train = open_data('data/training_data/ex2_training_data.txt')
ex2_test = open_data('data/testing_data/ex2_testing_data.txt')

# Experiment 3
ex3_train = open_data('data/training_data/ex3_training_data.txt')
ex3_test = open_data('data/testing_data/ex3_testing_data.txt')

# Experiment 4
ex4_train = open_data('data/training_data/ex4_training_data.txt')
ex4_test = open_data('data/testing_data/ex4_testing_data.txt')

# Experiment 5
ex5_train = open_data('data/training_data/ex5_training_data.txt')
ex5_test = open_data('data/testing_data/ex5_testing_data.txt')

# Experiment 6
ex6_train = open_data('data/training_data/ex6_training_data.txt')
ex6_test = open_data('data/testing_data/ex6_testing_data.txt')

# Train the model on all experiment training data
training_data_inputs = ex1_train + ex2_train + ex3_train + ex4_train + ex5_train + ex6_train
training_data_results = calculate_model_outputs(training_data_inputs)
print("***** Training Model *****")
print(training_data_results)

# Aggregate all test data and compare results to model outputs
test_data_inputs = ex1_test + ex2_test + ex3_test + ex4_test + ex5_test + ex6_test
test_data_results = calculate_model_outputs(test_data_inputs)
print("***** Testing Model *****")
print(test_data_results)

# Find the model difference of the training data vs testing data
experiment_data_diff = training_data_results - test_data_results
experiment_data_diff = experiment_data_diff.abs()
print("***** Training Model vs Testing Model *****")
print(experiment_data_diff)

# Calculate percent error
test_minus_train = test_data_results.sub(training_data_results).abs()
pct_error = test_minus_train.div(training_data_results).mul(100)
print("***** Percent Error *****")
print(pct_error)

# Create rankings of norms based on contextual factors, and then compare trainng and testing rankings
# training_rankings = create_ranking_matrix(training_data_results)
# testing_rankings = create_ranking_matrix(test_data_results)
# compare_rankings(training_rankings, testing_rankings)
# IA is the only cf with different rankings (train: D>P>B, test: B>P>D)

''' Model Evaluation: 5 x 2 Cross-Validation '''
# Take the entire dataset and split (randomly) half training and half testing data (done in split_data.py)
# Then, calculate the average distance from the training results to the testing results

cv_pct_error = pd.DataFrame(np.zeros(shape=(8, 4)), columns=['contextual_factor', 'D', 'B', 'P'])
cv_pct_error['contextual_factor'] = ['PFH,IA,TP', 'PFH,IA', 'PFH,TP', 'IA,TP', 'PFH', 'IA', 'TP', 'None']
cv_pct_error = cv_pct_error.set_index('contextual_factor', drop=True)

cv0_train = open_data('data/training_data/cv1_training_data.txt')
cv0_test = open_data('data/testing_data/cv1_testing_data.txt')
cv0_train_results = calculate_model_outputs(cv0_train)
cv0_test_results = calculate_model_outputs(cv0_test)
cv0_dist_to_train = cv0_test_results.sub(cv0_train_results).abs()
cv0_error = cv0_dist_to_train.div(cv0_train_results).mul(100)
cv_pct_error = cv_pct_error + cv0_error

cv1_train = open_data('data/training_data/cv2_training_data.txt')
cv1_test = open_data('data/testing_data/cv2_testing_data.txt')
cv1_train_results = calculate_model_outputs(cv1_train)
cv1_test_results = calculate_model_outputs(cv1_test)
cv1_dist_to_train = cv1_test_results.sub(cv1_train_results).abs()
cv1_error = cv1_dist_to_train.div(cv1_train_results).mul(100)
cv_pct_error = cv_pct_error + cv1_error

cv2_train = open_data('data/training_data/cv3_training_data.txt')
cv2_test = open_data('data/testing_data/cv3_testing_data.txt')
cv2_train_results = calculate_model_outputs(cv2_train)
cv2_test_results = calculate_model_outputs(cv2_test)
cv2_dist_to_train = cv2_test_results.sub(cv2_train_results).abs()
cv2_error = cv2_dist_to_train.div(cv2_train_results).mul(100)
cv_pct_error = cv_pct_error + cv2_error

cv3_train = open_data('data/training_data/cv4_training_data.txt')
cv3_test = open_data('data/testing_data/cv4_testing_data.txt')
cv3_train_results = calculate_model_outputs(cv3_train)
cv3_test_results = calculate_model_outputs(cv3_test)
cv3_dist_to_train = cv3_test_results.sub(cv3_train_results).abs()
cv3_error = cv3_dist_to_train.div(cv3_train_results).mul(100)
cv_pct_error = cv_pct_error + cv3_error

cv4_train = open_data('data/training_data/cv5_training_data.txt')
cv4_test = open_data('data/testing_data/cv5_testing_data.txt')
cv4_train_results = calculate_model_outputs(cv4_train)
cv4_test_results = calculate_model_outputs(cv4_test)
cv4_dist_to_train = cv4_test_results.sub(cv4_train_results).abs()
cv4_error = cv4_dist_to_train.div(cv4_train_results).mul(100)
cv_pct_error = cv_pct_error + cv4_error

# Find the average percent error
ave_errors = cv_pct_error.div(5)
print('\n')
print("Average Errors:")
print(ave_errors)
print('\n')

# # Find the average CPT from all of the 5 CVs
# all_train_cpts = cv1_train_results + cv2_train_results + cv3_train_results + cv4_train_results + cv0_train_results
# ave_cpt = all_train_cpts .div(5)
# print('\n')
# print("Average CPTs:")
# print(ave_cpt)
# print('\n')

'''
# Find the abs((distance from training - testing) - (ave. distances of 5 x 2 CV))
model_dist_minus_cv_dist = experiment_data_diff - ave_distances
model_dist_minus_cv_dist = model_dist_minus_cv_dist.abs()
print("abs((distance from training - testing) - (ave. distances of 5 x 2 CV)):")
print(model_dist_minus_cv_dist)
print('\n')

all_sums = []
all_sums_dict = {}
for index, row in ave_distances.iterrows():
    row_sum = np.sum(row.values)
    all_sums.append(row_sum)
    all_sums_dict[row_sum] = index
    # print(index, row_sum)

all_sums.sort()

# Find the corresponding contextual factor
for sum in all_sums:
    print(all_sums_dict.get(sum))

# Rankings for the average distances.
# Note that the smaller the number, the better, so look at 3,2,1
distance_rankings = create_ranking_matrix(ave_distances)
# print(distance_rankings)

# See the rankings for the distance between the normal training and testing.
# Note that the smaller the number, the better, so look at 3,2,1
distance_rankings_norm = create_ranking_matrix(experiment_data_diff)
# print(distance_rankings_norm)
'''

''' Additional Model Evaluation Techniques Below '''

"""
''' Model Evaluation: LOOCV - leave one experiment out '''
# print("***** Performing LOOCV Model Evaluation *****")

ex1_data = open_data('data/experiment_data/ex1_data.txt')
ex2_data = open_data('data/experiment_data/ex2_data.txt')
ex3_data = open_data('data/experiment_data/ex3_data.txt')
ex4_data = open_data('data/experiment_data/ex4_data.txt')
ex5_data = open_data('data/experiment_data/ex5_data.txt')
ex6_data = open_data('data/experiment_data/ex6_data.txt')

# Leave first experiment out and compare to mode
loocv_data = ex2_data + ex3_data + ex4_data + ex5_data + ex6_data
loocv_outputs = calculate_model_outputs(loocv_data)
loocv_diff = training_data_results - loocv_outputs
# print(loocv_diff)

# Leave second experiment out and compare to model
loocv_data = ex1_data + ex3_data + ex4_data + ex5_data + ex6_data
loocv_outputs = calculate_model_outputs(loocv_data)
loocv_diff = training_data_results - loocv_outputs
# print(loocv_diff)

# Leave third experiment out and compare to model
loocv_data = ex1_data + ex2_data + ex4_data + ex5_data + ex6_data
loocv_outputs = calculate_model_outputs(loocv_data)
loocv_diff = training_data_results - loocv_outputs
# print(loocv_diff)

# Leave fourth experiment out and compare to model
loocv_data = ex1_data + ex2_data + ex3_data + ex5_data + ex6_data
loocv_outputs = calculate_model_outputs(loocv_data)
loocv_diff = training_data_results - loocv_outputs
# print(loocv_diff)

# Leave fifth experiment out and compare to model
loocv_data = ex1_data + ex2_data + ex3_data + ex4_data + ex6_data
loocv_outputs = calculate_model_outputs(loocv_data)
loocv_diff = training_data_results - loocv_outputs
# print(loocv_diff)

# Leave sixth experiment out and compare to model
loocv_data = ex1_data + ex2_data + ex3_data + ex4_data + ex5_data
loocv_outputs = calculate_model_outputs(loocv_data)
loocv_diff = training_data_results - loocv_outputs
# print(loocv_diff)

''' Model Evaluation: k-fold cross-validation (k = 6) '''
# print("***** Performing K-Fold CV Model Evaluation (k=6) *****")

experiment_data = ex1_data + ex2_data + ex3_data + ex4_data + ex5_data + ex6_data
experiment_data = np.array(experiment_data)

# Split data into 6 random subsets
kf = KFold(n_splits=6, shuffle=True)
kf.get_n_splits(experiment_data)

all_differences = pd.DataFrame(np.zeros(shape=(8, 4)), columns=['contextual_factor', 'D', 'B', 'P'])
all_differences['contextual_factor'] = ['PFH,IA,TP', 'PFH,IA', 'PFH,TP', 'IA,TP', 'PFH', 'IA', 'TP', 'None']
all_differences = all_differences.set_index('contextual_factor', drop=True)

for train_index, test_index in kf.split(experiment_data):
    # print("TRAIN:", len(train_index), "TEST:", len(test_index))
    ex_train = experiment_data[train_index]
    ex_test = experiment_data[test_index]

    ex_train_outputs = calculate_model_outputs(ex_train)
    ex_test_outputs = calculate_model_outputs(ex_test)
    ex_diff = ex_train_outputs - ex_test_outputs

    all_differences = all_differences + ex_diff

all_differences = all_differences.div(6)
# print(all_differences)


'''
# Use this code below to compare models.
# Now compare all differences to the model differences
print(experiment_data_diff)

# Find which model is more accurate
testing_ex_model = all_differences - experiment_data_diff
print(testing_ex_model)

test_i = 0
all_i = 0

for col in testing_ex_model:
    for cf in testing_ex_model[col].index:
        test_value_ex = abs(testing_ex_model.loc[cf, col])
        test_value_all = abs(testing_all_model.loc[cf, col])

        if test_value_ex < test_value_all:
            test_i += 1
        else:
            all_i += 1

print("Experiment successes: ", test_i)  # 15/24
print("All successes: ", all_i)  # 9/24
'''

'''
# Finding KL divergence
kl_div_0 = stats.entropy(pk=training_data_results.loc['PFH,IA,TP'].values, qk=test_data_results.loc['PFH,IA,TP'].values)
print(kl_div_0)

kl_div_1 = stats.entropy(pk=training_data_results.loc['PFH,IA'].values, qk=test_data_results.loc['PFH,IA'].values)
print(kl_div_1)

kl_div_2 = stats.entropy(pk=training_data_results.loc['PFH,TP'].values, qk=test_data_results.loc['PFH,TP'].values)
print(kl_div_2)

kl_div_3 = stats.entropy(pk=training_data_results.loc['IA,TP'].values, qk=test_data_results.loc['IA,TP'].values)
print(kl_div_3)

kl_div_4 = stats.entropy(pk=training_data_results.loc['PFH'].values, qk=test_data_results.loc['PFH'].values)
print(kl_div_4)

kl_div_5 = stats.entropy(pk=training_data_results.loc['IA'].values, qk=test_data_results.loc['IA'].values)
print(kl_div_5)

kl_div_6 = stats.entropy(pk=training_data_results.loc['TP'].values, qk=test_data_results.loc['TP'].values)
print(kl_div_6)

kl_div_7 = stats.entropy(pk=training_data_results.loc['None'].values, qk=test_data_results.loc['None'].values)
print(kl_div_7)
'''
"""
