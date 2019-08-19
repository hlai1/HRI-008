"""
This is a program for calculating the stats from the experiments.
The per-person stats and conditional probabilities can be used for model evaluation.

Probability Distribution Table 

(Used for model evaluation.)

The following probabilities are calculated from the experimental data:
(These are calculated per-person, per-experiment, and per-total data 
to examine consistency within players, sessions, and the experimental design as a whole.)

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


def open_file(filename):
    with open(filename, "r"):
        lines = [p.split() for p in [line.rstrip('\n') for line in open(filename)]]

    return lines


def calculate_per_person_probabilities(experiment_lines, player_id, dimensions, contextual_factors, norms):

    # Create a new dataframe for the probability distribution
    player_probs = pd.DataFrame(dimensions, columns=norms)
    player_probs['contextual_factor'] = contextual_factors
    player_probs = player_probs.set_index('contextual_factor', drop=True)

    # Initialize the total number of utterances per contextual factor
    total_pfh_ia_tp, total_pfh_ia, total_pfh_tp, total_ia_tp, total_pfh, total_ia, total_tp, total_none = 0, 0, 0, 0, 0, 0, 0, 0

    # Initialize the total number of utterances per norm: D
    d_pfh_ia_tp, d_pfh_ia, d_pfh_tp, d_ia_tp, d_pfh, d_ia, d_tp, d_none = 0, 0, 0, 0, 0, 0, 0, 0

    # Initialize the total number of utterances per norm: B
    b_pfh_ia_tp, b_pfh_ia, b_pfh_tp, b_ia_tp, b_pfh, b_ia, b_tp, b_none = 0, 0, 0, 0, 0, 0, 0, 0

    # Initialize the total number of utterances per norm: P
    p_pfh_ia_tp, p_pfh_ia, p_pfh_tp, p_ia_tp, p_pfh, p_ia, p_tp, p_none = 0, 0, 0, 0, 0, 0, 0, 0

    # Populate the dataframe with the player's annotations
    for line in experiment_lines:
        if len(line) > 7:  # the min len of the lines we want to consider

            player = line[7][1]  # Player ID
            norm = line[6][1]  # D/B/P
            ia = line[1][0]  # IA: Y or N
            pfh = line[3][0]  # PFH: Y or N
            tp = line[5][0]  # TP: Y or N

            if player == player_id:
                if norm == 'D':
                    if pfh == 'Y' and ia == 'Y' and tp == 'Y':
                        d_pfh_ia_tp += 1
                        total_pfh_ia_tp += 1
                    elif pfh == 'Y' and ia == 'Y' and tp == 'N':
                        d_pfh_ia += 1
                        total_pfh_ia += 1
                    elif pfh == 'Y' and ia == 'N' and tp == 'Y':
                        d_pfh_tp += 1
                        total_pfh_tp += 1
                    elif pfh == 'N' and ia == 'Y' and tp == 'Y':
                        d_ia_tp += 1
                        total_ia_tp += 1
                    elif pfh == 'Y' and ia == 'N' and tp == 'N':
                        d_pfh += 1
                        total_pfh += 1
                    elif pfh == 'N' and ia == 'Y' and tp == 'N':
                        d_ia += 1
                        total_ia += 1
                    elif pfh == 'N' and ia == 'N' and tp == 'Y':
                        d_tp += 1
                        total_tp += 1
                    elif pfh == 'N' and ia == 'N' and tp == 'N':
                        d_none += 1
                        total_none += 1

                elif norm == 'B':
                    if pfh == 'Y' and ia == 'Y' and tp == 'Y':
                        b_pfh_ia_tp += 1
                        total_pfh_ia_tp += 1
                    elif pfh == 'Y' and ia == 'Y' and tp == 'N':
                        b_pfh_ia += 1
                        total_pfh_ia += 1
                    elif pfh == 'Y' and ia == 'N' and tp == 'Y':
                        b_pfh_tp += 1
                        total_pfh_tp += 1
                    elif pfh == 'N' and ia == 'Y' and tp == 'Y':
                        b_ia_tp += 1
                        total_ia_tp += 1
                    elif pfh == 'Y' and ia == 'N' and tp == 'N':
                        b_pfh += 1
                        total_pfh += 1
                    elif pfh == 'N' and ia == 'Y' and tp == 'N':
                        b_ia += 1
                        total_ia += 1
                    elif pfh == 'N' and ia == 'N' and tp == 'Y':
                        b_tp += 1
                        total_tp += 1
                    elif pfh == 'N' and ia == 'N' and tp == 'N':
                        b_none += 1
                        total_none += 1

                elif norm == 'P':
                    if pfh == 'Y' and ia == 'Y' and tp == 'Y':
                        p_pfh_ia_tp += 1
                        total_pfh_ia_tp += 1
                    elif pfh == 'Y' and ia == 'Y' and tp == 'N':
                        p_pfh_ia += 1
                        total_pfh_ia += 1
                    elif pfh == 'Y' and ia == 'N' and tp == 'Y':
                        p_pfh_tp += 1
                        total_pfh_tp += 1
                    elif pfh == 'N' and ia == 'Y' and tp == 'Y':
                        p_ia_tp += 1
                        total_ia_tp += 1
                    elif pfh == 'Y' and ia == 'N' and tp == 'N':
                        p_pfh += 1
                        total_pfh += 1
                    elif pfh == 'N' and ia == 'Y' and tp == 'N':
                        p_ia += 1
                        total_ia += 1
                    elif pfh == 'N' and ia == 'N' and tp == 'Y':
                        p_tp += 1
                        total_tp += 1
                    elif pfh == 'N' and ia == 'N' and tp == 'N':
                        p_none += 1
                        total_none += 1

    if total_pfh_ia_tp > 0:
        player_probs['D']['PFH,IA,TP'] = d_pfh_ia_tp / total_pfh_ia_tp
        player_probs['B']['PFH,IA,TP'] = b_pfh_ia_tp / total_pfh_ia_tp
        player_probs['P']['PFH,IA,TP'] = p_pfh_ia_tp / total_pfh_ia_tp
    else:
        player_probs['D']['PFH,IA,TP'] = np.nan
        player_probs['B']['PFH,IA,TP'] = np.nan
        player_probs['P']['PFH,IA,TP'] = np.nan

    if total_pfh_ia > 0:
        player_probs['D']['PFH,IA'] = d_pfh_ia / total_pfh_ia
        player_probs['B']['PFH,IA'] = b_pfh_ia / total_pfh_ia
        player_probs['P']['PFH,IA'] = p_pfh_ia / total_pfh_ia
    else:
        player_probs['D']['PFH,IA'] = np.nan
        player_probs['B']['PFH,IA'] = np.nan
        player_probs['P']['PFH,IA'] = np.nan

    if total_pfh_tp > 0:
        player_probs['D']['PFH,TP'] = d_pfh_tp / total_pfh_tp
        player_probs['B']['PFH,TP'] = b_pfh_tp / total_pfh_tp
        player_probs['P']['PFH,TP'] = p_pfh_tp / total_pfh_tp
    else:
        player_probs['D']['PFH,TP'] = np.nan
        player_probs['B']['PFH,TP'] = np.nan
        player_probs['P']['PFH,TP'] = np.nan

    if total_ia_tp > 0:
        player_probs['D']['IA,TP'] = d_ia_tp / total_ia_tp
        player_probs['B']['IA,TP'] = b_ia_tp / total_ia_tp
        player_probs['P']['IA,TP'] = p_ia_tp / total_ia_tp
    else:
        player_probs['D']['IA,TP'] = np.nan
        player_probs['B']['IA,TP'] = np.nan
        player_probs['P']['IA,TP'] = np.nan

    if total_pfh > 0:
        player_probs['D']['PFH'] = d_pfh / total_pfh
        player_probs['B']['PFH'] = b_pfh / total_pfh
        player_probs['P']['PFH'] = p_pfh / total_pfh
    else:
        player_probs['D']['PFH'] = np.nan
        player_probs['B']['PFH'] = np.nan
        player_probs['P']['PFH'] = np.nan

    if total_ia > 0:
        player_probs['D']['IA'] = d_ia / total_ia
        player_probs['B']['IA'] = b_ia / total_ia
        player_probs['P']['IA'] = p_ia / total_ia
    else:
        player_probs['D']['IA'] = np.nan
        player_probs['B']['IA'] = np.nan
        player_probs['P']['IA'] = np.nan

    if total_tp > 0:
        player_probs['D']['TP'] = d_tp / total_tp
        player_probs['B']['TP'] = b_tp / total_tp
        player_probs['P']['TP'] = p_tp / total_tp
    else:
        player_probs['D']['TP'] = np.nan
        player_probs['B']['TP'] = np.nan
        player_probs['P']['TP'] = np.nan

    if total_none > 0:
        player_probs['D']['None'] = d_none / total_none
        player_probs['B']['None'] = b_none / total_none
        player_probs['P']['None'] = p_none / total_none
    else:
        player_probs['D']['None'] = np.nan
        player_probs['B']['None'] = np.nan
        player_probs['P']['None'] = np.nan

    return player_probs


def calculate_experiment_probabilities(experiment_lines, dimensions, contextual_factors, norms):

    # Create a new dataframe for the probability distribution
    experiment_probs = pd.DataFrame(dimensions, columns=norms)
    experiment_probs['contextual_factor'] = contextual_factors
    experiment_probs = experiment_probs.set_index('contextual_factor', drop=True)

    # Initialize the total number of utterances per contextual factor
    total_pfh_ia_tp, total_pfh_ia, total_pfh_tp, total_ia_tp, total_pfh, total_ia, total_tp, total_none = 0, 0, 0, 0, 0, 0, 0, 0

    # Initialize the total number of utterances per norm: D
    d_pfh_ia_tp, d_pfh_ia, d_pfh_tp, d_ia_tp, d_pfh, d_ia, d_tp, d_none = 0, 0, 0, 0, 0, 0, 0, 0

    # Initialize the total number of utterances per norm: B
    b_pfh_ia_tp, b_pfh_ia, b_pfh_tp, b_ia_tp, b_pfh, b_ia, b_tp, b_none = 0, 0, 0, 0, 0, 0, 0, 0

    # Initialize the total number of utterances per norm: P
    p_pfh_ia_tp, p_pfh_ia, p_pfh_tp, p_ia_tp, p_pfh, p_ia, p_tp, p_none = 0, 0, 0, 0, 0, 0, 0, 0

    # Populate the dataframe with the player's annotations
    for line in experiment_lines:
        if len(line) > 6:  # the min len of the lines we want to consider

            norm = line[6][1]  # D/B/P
            ia = line[1][0]  # IA: Y or N
            pfh = line[3][0]  # PFH: Y or N
            tp = line[5][0]  # TP: Y or N

            if norm == 'D':
                if pfh == 'Y' and ia == 'Y' and tp == 'Y':
                    d_pfh_ia_tp += 1
                    total_pfh_ia_tp += 1
                elif pfh == 'Y' and ia == 'Y' and tp == 'N':
                    d_pfh_ia += 1
                    total_pfh_ia += 1
                elif pfh == 'Y' and ia == 'N' and tp == 'Y':
                    d_pfh_tp += 1
                    total_pfh_tp += 1
                elif pfh == 'N' and ia == 'Y' and tp == 'Y':
                    d_ia_tp += 1
                    total_ia_tp += 1
                elif pfh == 'Y' and ia == 'N' and tp == 'N':
                    d_pfh += 1
                    total_pfh += 1
                elif pfh == 'N' and ia == 'Y' and tp == 'N':
                    d_ia += 1
                    total_ia += 1
                elif pfh == 'N' and ia == 'N' and tp == 'Y':
                    d_tp += 1
                    total_tp += 1
                elif pfh == 'N' and ia == 'N' and tp == 'N':
                    d_none += 1
                    total_none += 1

            elif norm == 'B':
                if pfh == 'Y' and ia == 'Y' and tp == 'Y':
                    b_pfh_ia_tp += 1
                    total_pfh_ia_tp += 1
                elif pfh == 'Y' and ia == 'Y' and tp == 'N':
                    b_pfh_ia += 1
                    total_pfh_ia += 1
                elif pfh == 'Y' and ia == 'N' and tp == 'Y':
                    b_pfh_tp += 1
                    total_pfh_tp += 1
                elif pfh == 'N' and ia == 'Y' and tp == 'Y':
                    b_ia_tp += 1
                    total_ia_tp += 1
                elif pfh == 'Y' and ia == 'N' and tp == 'N':
                    b_pfh += 1
                    total_pfh += 1
                elif pfh == 'N' and ia == 'Y' and tp == 'N':
                    b_ia += 1
                    total_ia += 1
                elif pfh == 'N' and ia == 'N' and tp == 'Y':
                    b_tp += 1
                    total_tp += 1
                elif pfh == 'N' and ia == 'N' and tp == 'N':
                    b_none += 1
                    total_none += 1

            elif norm == 'P':
                if pfh == 'Y' and ia == 'Y' and tp == 'Y':
                    p_pfh_ia_tp += 1
                    total_pfh_ia_tp += 1
                elif pfh == 'Y' and ia == 'Y' and tp == 'N':
                    p_pfh_ia += 1
                    total_pfh_ia += 1
                elif pfh == 'Y' and ia == 'N' and tp == 'Y':
                    p_pfh_tp += 1
                    total_pfh_tp += 1
                elif pfh == 'N' and ia == 'Y' and tp == 'Y':
                    p_ia_tp += 1
                    total_ia_tp += 1
                elif pfh == 'Y' and ia == 'N' and tp == 'N':
                    p_pfh += 1
                    total_pfh += 1
                elif pfh == 'N' and ia == 'Y' and tp == 'N':
                    p_ia += 1
                    total_ia += 1
                elif pfh == 'N' and ia == 'N' and tp == 'Y':
                    p_tp += 1
                    total_tp += 1
                elif pfh == 'N' and ia == 'N' and tp == 'N':
                    p_none += 1
                    total_none += 1

    if total_pfh_ia_tp > 0:
        experiment_probs['D']['PFH,IA,TP'] = d_pfh_ia_tp / total_pfh_ia_tp
        experiment_probs['B']['PFH,IA,TP'] = b_pfh_ia_tp / total_pfh_ia_tp
        experiment_probs['P']['PFH,IA,TP'] = p_pfh_ia_tp / total_pfh_ia_tp
    else:
        experiment_probs['D']['PFH,IA,TP'] = np.nan
        experiment_probs['B']['PFH,IA,TP'] = np.nan
        experiment_probs['P']['PFH,IA,TP'] = np.nan

    if total_pfh_ia > 0:
        experiment_probs['D']['PFH,IA'] = d_pfh_ia / total_pfh_ia
        experiment_probs['B']['PFH,IA'] = b_pfh_ia / total_pfh_ia
        experiment_probs['P']['PFH,IA'] = p_pfh_ia / total_pfh_ia
    else:
        experiment_probs['D']['PFH,IA'] = np.nan
        experiment_probs['B']['PFH,IA'] = np.nan
        experiment_probs['P']['PFH,IA'] = np.nan

    if total_pfh_tp > 0:
        experiment_probs['D']['PFH,TP'] = d_pfh_tp / total_pfh_tp
        experiment_probs['B']['PFH,TP'] = b_pfh_tp / total_pfh_tp
        experiment_probs['P']['PFH,TP'] = p_pfh_tp / total_pfh_tp
    else:
        experiment_probs['D']['PFH,TP'] = np.nan
        experiment_probs['B']['PFH,TP'] = np.nan
        experiment_probs['P']['PFH,TP'] = np.nan

    if total_ia_tp > 0:
        experiment_probs['D']['IA,TP'] = d_ia_tp / total_ia_tp
        experiment_probs['B']['IA,TP'] = b_ia_tp / total_ia_tp
        experiment_probs['P']['IA,TP'] = p_ia_tp / total_ia_tp
    else:
        experiment_probs['D']['IA,TP'] = np.nan
        experiment_probs['B']['IA,TP'] = np.nan
        experiment_probs['P']['IA,TP'] = np.nan

    if total_pfh > 0:
        experiment_probs['D']['PFH'] = d_pfh / total_pfh
        experiment_probs['B']['PFH'] = b_pfh / total_pfh
        experiment_probs['P']['PFH'] = p_pfh / total_pfh
    else:
        experiment_probs['D']['PFH'] = np.nan
        experiment_probs['B']['PFH'] = np.nan
        experiment_probs['P']['PFH'] = np.nan

    if total_ia > 0:
        experiment_probs['D']['IA'] = d_ia / total_ia
        experiment_probs['B']['IA'] = b_ia / total_ia
        experiment_probs['P']['IA'] = p_ia / total_ia
    else:
        experiment_probs['D']['IA'] = np.nan
        experiment_probs['B']['IA'] = np.nan
        experiment_probs['P']['IA'] = np.nan

    if total_tp > 0:
        experiment_probs['D']['TP'] = d_tp / total_tp
        experiment_probs['B']['TP'] = b_tp / total_tp
        experiment_probs['P']['TP'] = p_tp / total_tp
    else:
        experiment_probs['D']['TP'] = np.nan
        experiment_probs['B']['TP'] = np.nan
        experiment_probs['P']['TP'] = np.nan

    if total_none > 0:
        experiment_probs['D']['None'] = d_none / total_none
        experiment_probs['B']['None'] = b_none / total_none
        experiment_probs['P']['None'] = p_none / total_none
    else:
        experiment_probs['D']['None'] = np.nan
        experiment_probs['B']['None'] = np.nan
        experiment_probs['P']['None'] = np.nan

    return experiment_probs


experiment_1_lines = open_file('data/experiment_scripts/ex1_script.txt')
experiment_2_lines = open_file('data/experiment_scripts/ex2_script.txt')

df_shape = np.zeros(shape=(8, 4))
contextual_factors = ['PFH,IA,TP', 'PFH,IA', 'PFH,TP', 'IA,TP', 'PFH', 'IA', 'TP', 'None']
norms = ['contextual_factor', 'D', 'B', 'P']

'''Per-Participant Probabilities: Experiment 1 (Time Pressure)'''
player_1_ex_1_probs = calculate_per_person_probabilities(experiment_1_lines, 'G', df_shape, contextual_factors, norms)
print("Player 1, Experiment 1 Probabilities: ")
print(player_1_ex_1_probs)
print('\n')

player_2_ex_1_probs = calculate_per_person_probabilities(experiment_1_lines, 'B', df_shape, contextual_factors, norms)
print("Player 2, Experiment 1 Probabilities: ")
print(player_2_ex_1_probs)
print('\n')

player_3_ex_1_probs = calculate_per_person_probabilities(experiment_1_lines, 'R', df_shape, contextual_factors, norms)
print("Player 3, Experiment 1 Probabilities: ")
print(player_3_ex_1_probs)
print('\n')

'''Per-Participant Probabilities: Experiment 2 (No Time Pressure)'''
player_1_ex_2_probs = calculate_per_person_probabilities(experiment_2_lines, 'B', df_shape, contextual_factors, norms)
print("Player 1, Experiment 2 Probabilities: ")
print(player_1_ex_2_probs)
print('\n')

player_2_ex_2_probs = calculate_per_person_probabilities(experiment_2_lines, 'Y', df_shape, contextual_factors, norms)
print("Player 2, Experiment 2 Probabilities: ")
print(player_2_ex_2_probs)
print('\n')

player_3_ex_2_probs = calculate_per_person_probabilities(experiment_2_lines, 'W', df_shape, contextual_factors, norms)
print("Player 3, Experiment 2 Probabilities: ")
print(player_3_ex_2_probs)
print('\n')

'''Per-Experiment Probabilities: Experiment 1 (Time Pressure)'''
experiment_1_probs = calculate_experiment_probabilities(experiment_1_lines, df_shape, contextual_factors, norms)
print("Experiment 1 Probabilities: ")
print(experiment_1_probs)
print('\n')

'''Per-Experiment Probabilities: Experiment 2 (No Time Pressure)'''
experiment_2_probs = calculate_experiment_probabilities(experiment_2_lines, df_shape, contextual_factors, norms)
print("Experiment 2 Probabilities: ")
print(experiment_2_probs)
print('\n')

'''All-Experiment Probabilities'''
all_experiment_lines = experiment_1_lines + experiment_2_lines
all_experiment_probs = calculate_experiment_probabilities(all_experiment_lines, df_shape, contextual_factors, norms)
print("All Probabilities: ")
print(all_experiment_probs)