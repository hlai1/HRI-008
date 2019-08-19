import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
# import matplotlib.pyplot as plt


def open_file(filename, player_id):
    with open(filename, "r"):
        lines = [p.split() for p in [line.rstrip('\n') for line in open(filename)]]

        wcs = []

        for line in lines:
            # For annotations that already have the final format:
            if len(line) > 7:
                if line[7][1] == player_id:
                    wc = len(line) - 8
                    wcs.append(wc)

    return wcs


def calc_word_count_stats(unsorted_wc):
    unsorted_wc.sort()
    wcs = np.array(unsorted_wc)
    uniques, counts = np.unique(wcs, return_counts=True)
    median = np.median(wcs)
    total = len(wcs)

    return uniques, counts, median, total


''' Experiment 1 Brevity Stats'''
p1_ex_1_wc = open_file('data/experiment_scripts/ex1_script.txt', 'G')
p2_ex_1_wc = open_file('data/experiment_scripts/ex1_script.txt', 'B')
p3_ex_1_wc = open_file('data/experiment_scripts/ex1_script.txt', 'R')

''' Player 1 '''
p1_ex_1_uniques, p1_ex_1_counts, p1_ex_1_median, p1_ex_1_total = calc_word_count_stats(p1_ex_1_wc)
print("Experiment 1, Player 1:")
print("Frequencies: ", dict(zip(p1_ex_1_uniques, p1_ex_1_counts)))
print("Median: ", p1_ex_1_median, "Total: ", p1_ex_1_total)

''' Player 2 '''
p2_ex_1_uniques, p2_ex_1_counts, p2_ex_1_median, p2_ex_1_total = calc_word_count_stats(p2_ex_1_wc)
print("Experiment 1, Player 2:")
print("Frequencies: ", dict(zip(p2_ex_1_uniques, p2_ex_1_counts)))
print("Median: ", p2_ex_1_median, "Total: ", p2_ex_1_total)

''' Player 3 '''
p3_ex_1_uniques, p3_ex_1_counts, p3_ex_1_median, p3_ex_1_total = calc_word_count_stats(p3_ex_1_wc)
print("Experiment 1, Player 3:")
print("Frequencies: ", dict(zip(p3_ex_1_uniques, p3_ex_1_counts)))
print("Median: ", p3_ex_1_median, "Total: ", p3_ex_1_total)

''' Overall - Experiment 1 '''
total_wcs_ex_1 = p1_ex_1_wc + p2_ex_1_wc + p3_ex_1_wc
total_ex_1_uniques, total_ex_1_counts, total_ex_1_median, total_ex_1_total = calc_word_count_stats(total_wcs_ex_1)
print("Experiment 1 (Overall):")
print("Frequencies: ", dict(zip(total_ex_1_uniques, total_ex_1_counts)))
print("Median: ", total_ex_1_median, "Total: ", total_ex_1_total)
print('\n')

''' Experiment 2 Brevity Stats '''
p1_ex_2_wc = open_file('data/experiment_scripts/ex2_script.txt', 'B')
p2_ex_2_wc = open_file('data/experiment_scripts/ex2_script.txt', 'Y')
p3_ex_2_wc = open_file('data/experiment_scripts/ex2_script.txt', 'W')

''' Player 1 '''
p1_ex_2_uniques, p1_ex_2_counts, p1_ex_2_median, p1_ex_2_total = calc_word_count_stats(p1_ex_2_wc)
print("Experiment 2, Player 1:")
print("Frequencies: ", dict(zip(p1_ex_2_uniques, p1_ex_2_counts)))
print("Median: ", p1_ex_2_median, "Total: ", p1_ex_2_total)

''' Player 2 '''
p2_ex_2_uniques, p2_ex_2_counts, p2_ex_2_median, p2_ex_2_total = calc_word_count_stats(p2_ex_2_wc)
print("Experiment 2, Player 2:")
print("Frequencies: ", dict(zip(p2_ex_2_uniques, p2_ex_2_counts)))
print("Median: ", p2_ex_2_median, "Total: ", p2_ex_2_total)

''' Player 3 '''
p3_ex_2_uniques, p3_ex_2_counts, p3_ex_2_median, p3_ex_2_total = calc_word_count_stats(p3_ex_2_wc)
print("Experiment 2, Player 3:")
print("Frequencies: ", dict(zip(p3_ex_2_uniques, p3_ex_2_counts)))
print("Median: ", p3_ex_2_median, "Total: ", p3_ex_2_total)

''' Overall - Experiment 2 '''
total_wcs_ex_2 = p1_ex_2_wc + p2_ex_2_wc + p3_ex_2_wc
total_ex_2_uniques, total_ex_2_counts, total_ex_2_median, total_ex_2_total = calc_word_count_stats(total_wcs_ex_2)
print("Experiment 2 (Overall):")
print("Frequencies: ", dict(zip(total_ex_2_uniques, total_ex_2_counts)))
print("Median: ", total_ex_2_median, "Total: ", total_ex_2_total)
print('\n')

''' Experiment 3 Brevity Stats '''
p1_ex_3_wc = open_file('data/experiment_scripts/ex3_script.txt', 'B')
p2_ex_3_wc = open_file('data/experiment_scripts/ex3_script.txt', 'G')
p3_ex_3_wc = open_file('data/experiment_scripts/ex3_script.txt', 'R')

''' Player 1 '''
p1_ex_3_uniques, p1_ex_3_counts, p1_ex_3_median, p1_ex_3_total = calc_word_count_stats(p1_ex_3_wc)
print("Experiment 3, Player 1:")
print("Frequencies: ", dict(zip(p1_ex_3_uniques, p1_ex_3_counts)))
print("Median: ", p1_ex_3_median, "Total: ", p1_ex_3_total)

''' Player 2 '''
p2_ex_3_uniques, p2_ex_3_counts, p2_ex_3_median, p2_ex_3_total = calc_word_count_stats(p2_ex_3_wc)
print("Experiment 3, Player 2:")
print("Frequencies: ", dict(zip(p2_ex_3_uniques, p2_ex_3_counts)))
print("Median: ", p2_ex_3_median, "Total: ", p2_ex_3_total)

''' Player 3 '''
p3_ex_3_uniques, p3_ex_3_counts, p3_ex_3_median, p3_ex_3_total = calc_word_count_stats(p3_ex_3_wc)
print("Experiment 3, Player 3:")
print("Frequencies: ", dict(zip(p3_ex_3_uniques, p3_ex_3_counts)))
print("Median: ", p3_ex_3_median, "Total: ", p3_ex_3_total)

''' Overall - Experiment 3 '''
total_wcs_ex_3 = p1_ex_3_wc + p2_ex_3_wc + p3_ex_3_wc
total_ex_3_uniques, total_ex_3_counts, total_ex_3_median, total_ex_3_total = calc_word_count_stats(total_wcs_ex_3)
print("Experiment 3 (Overall):")
print("Frequencies: ", dict(zip(total_ex_3_uniques, total_ex_3_counts)))
print("Median: ", total_ex_3_median, "Total: ", total_ex_3_total)
print('\n')

''' Experiment 4 Brevity Stats '''
p1_ex_4_wc = open_file('data/experiment_scripts/ex4_script.txt', 'B')
p2_ex_4_wc = open_file('data/experiment_scripts/ex4_script.txt', 'G')
p3_ex_4_wc = open_file('data/experiment_scripts/ex4_script.txt', 'R')

''' Player 1 '''
p1_ex_4_uniques, p1_ex_4_counts, p1_ex_4_median, p1_ex_4_total = calc_word_count_stats(p1_ex_4_wc)
print("Experiment 4, Player 1:")
print("Frequencies: ", dict(zip(p1_ex_4_uniques, p1_ex_4_counts)))
print("Median: ", p1_ex_4_median, "Total: ", p1_ex_4_total)

''' Player 2 '''
p2_ex_4_uniques, p2_ex_4_counts, p2_ex_4_median, p2_ex_4_total = calc_word_count_stats(p2_ex_4_wc)
print("Experiment 4, Player 2:")
print("Frequencies: ", dict(zip(p2_ex_4_uniques, p2_ex_4_counts)))
print("Median: ", p2_ex_4_median, "Total: ", p2_ex_4_total)

''' Player 3 '''
p3_ex_4_uniques, p3_ex_4_counts, p3_ex_4_median, p3_ex_4_total = calc_word_count_stats(p3_ex_4_wc)
print("Experiment 4, Player 3:")
print("Frequencies: ", dict(zip(p3_ex_4_uniques, p3_ex_4_counts)))
print("Median: ", p3_ex_4_median, "Total: ", p3_ex_4_total)

''' Overall - Experiment 4 '''
total_wcs_ex_4 = p1_ex_4_wc + p2_ex_4_wc + p3_ex_4_wc
total_ex_4_uniques, total_ex_4_counts, total_ex_4_median, total_ex_4_total = calc_word_count_stats(total_wcs_ex_4)
print("Experiment 4 (Overall):")
print("Frequencies: ", dict(zip(total_ex_4_uniques, total_ex_4_counts)))
print("Median: ", total_ex_4_median, "Total: ", total_ex_4_total)
print('\n')

''' Experiment 5 Brevity Stats '''
p1_ex_5_wc = open_file('data/experiment_scripts/ex5_script.txt', 'B')
p2_ex_5_wc = open_file('data/experiment_scripts/ex5_script.txt', 'G')
p3_ex_5_wc = open_file('data/experiment_scripts/ex5_script.txt', 'R')

''' Player 1 '''
p1_ex_5_uniques, p1_ex_5_counts, p1_ex_5_median, p1_ex_5_total = calc_word_count_stats(p1_ex_5_wc)
print("Experiment 5, Player 1:")
print("Frequencies: ", dict(zip(p1_ex_5_uniques, p1_ex_5_counts)))
print("Median: ", p1_ex_5_median, "Total: ", p1_ex_5_total)

''' Player 2 '''
p2_ex_5_uniques, p2_ex_5_counts, p2_ex_5_median, p2_ex_5_total = calc_word_count_stats(p2_ex_5_wc)
print("Experiment 5, Player 2:")
print("Frequencies: ", dict(zip(p2_ex_5_uniques, p2_ex_5_counts)))
print("Median: ", p2_ex_5_median, "Total: ", p2_ex_5_total)

''' Player 3 '''
p3_ex_5_uniques, p3_ex_5_counts, p3_ex_5_median, p3_ex_5_total = calc_word_count_stats(p3_ex_5_wc)
print("Experiment 5, Player 3:")
print("Frequencies: ", dict(zip(p3_ex_5_uniques, p3_ex_5_counts)))
print("Median: ", p3_ex_5_median, "Total: ", p3_ex_5_total)

''' Overall - Experiment 5 '''
total_wcs_ex_5 = p1_ex_5_wc + p2_ex_5_wc + p3_ex_5_wc
total_ex_5_uniques, total_ex_5_counts, total_ex_5_median, total_ex_5_total = calc_word_count_stats(total_wcs_ex_5)
print("Experiment 5 (Overall):")
print("Frequencies: ", dict(zip(total_ex_5_uniques, total_ex_5_counts)))
print("Median: ", total_ex_5_median, "Total: ", total_ex_5_total)
print('\n')


''' Experiment 6 Brevity Stats '''
p1_ex_6_wc = open_file('data/experiment_scripts/ex6_script.txt', 'B')
p2_ex_6_wc = open_file('data/experiment_scripts/ex6_script.txt', 'G')
p3_ex_6_wc = open_file('data/experiment_scripts/ex6_script.txt', 'R')

''' Player 1 '''
p1_ex_6_uniques, p1_ex_6_counts, p1_ex_6_median, p1_ex_6_total = calc_word_count_stats(p1_ex_6_wc)
print("Experiment 6, Player 1:")
print("Frequencies: ", dict(zip(p1_ex_6_uniques, p1_ex_6_counts)))
print("Median: ", p1_ex_6_median, "Total: ", p1_ex_6_total)

''' Player 2 '''
p2_ex_6_uniques, p2_ex_6_counts, p2_ex_6_median, p2_ex_6_total = calc_word_count_stats(p2_ex_6_wc)
print("Experiment 6, Player 2:")
print("Frequencies: ", dict(zip(p2_ex_6_uniques, p2_ex_6_counts)))
print("Median: ", p2_ex_6_median, "Total: ", p2_ex_6_total)

''' Player 3 '''
p3_ex_6_uniques, p3_ex_6_counts, p3_ex_6_median, p3_ex_6_total = calc_word_count_stats(p3_ex_6_wc)
print("Experiment 6, Player 3:")
print("Frequencies: ", dict(zip(p3_ex_6_uniques, p3_ex_6_counts)))
print("Median: ", p3_ex_6_median, "Total: ", p3_ex_6_total)

''' Overall - Experiment 6 '''
total_wcs_ex_6 = p1_ex_6_wc + p2_ex_6_wc + p3_ex_6_wc
total_ex_6_uniques, total_ex_6_counts, total_ex_6_median, total_ex_6_total = calc_word_count_stats(total_wcs_ex_6)
print("Experiment 6 (Overall):")
print("Frequencies: ", dict(zip(total_ex_6_uniques, total_ex_6_counts)))
print("Median: ", total_ex_6_median, "Total: ", total_ex_6_total)
print('\n')

"""

Stats for Pilot Experiment:

51% of utterances have four words or less
40% of utterances have three words or less 
Conclusion: Brief = 4 words or less

Stats for Experiment 1:

46% of utterances have four words or less 
37% of utterances have three words or less
27% of utterances have two words or less
Conclusion: Brief = 3 words or less

Stats for Experiment 2:

42% of utterances have five words or less
35% of utterances have four words or less
27% utterances have three words or less
Conclusion: Brief = 4 words or less

Stats for Experiment 3:

42% of utterances have three words or less
34% of utterances have two words or less
Conclusion: Brief = 3 words or less

Stats for Experiment 4:

39% of utterances have five words or less
32% of utterances have four words or less 
Conclusion: Brief = 4 words or less 

Stats for Experiment 5:

46% of utterances have four words or less
32% of utterances have three words or less
Conclusion: Brief = 3 words or less 

Stats for Experiment 6:

40% of utterances have 4 words or less
32% of utterances have 3 words or less
24% of utterances have 2 words or less 
Conclusion: Brief = 3 words or less


"""

"""
# Plotting functions
p1_objects = [str(i) for i in p1_uniques]
y_pos = np.arange(len(p1_objects))

plt.bar(y_pos, p1_counts, align='center', alpha=0.5)
plt.xticks(y_pos, p1_objects)
plt.ylabel('Player 1 Frequency')
plt.xlabel('Player 1 Word Count')
plt.title('Player 1: Number of Words per Utterance')

plt.show()

p2_objects = [str(i) for i in p2_uniques]
y_pos = np.arange(len(p2_objects))

plt.bar(y_pos, p2_counts, align='center', alpha=0.5)
plt.xticks(y_pos, p2_objects)
plt.ylabel('Player 2 Frequency')
plt.xlabel('Player 2 Word Count')
plt.title('Player 2: Number of Words per Utterance')

plt.show()

p3_objects = [str(i) for i in p3_uniques]
y_pos = np.arange(len(p3_objects))

plt.bar(y_pos, p3_counts, align='center', alpha=0.5)
plt.xticks(y_pos, p3_objects)
plt.ylabel('Player 3 Frequency')
plt.xlabel('Player 3 Word Count')
plt.title('Player 3: Number of Words per Utterance')

plt.show()
"""
