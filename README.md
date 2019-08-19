# HRI-008
Mines MIRRORlab project to create a Bayesian network model that predicts communicative norms based on contextual factors.

## Experiment Annotation Process

After each experiment, a “script” is written of the dialogue from the video data (in a simple .txt file). All experiment scripts are located in data/experiment_scripts.
- Participant ID (to keep track of who says what)
- Norm (Direct/Brief/Polite)
- Brevity is calculated based on the frequencies of the utterance lengths per experiment (in brevity.py)
  - (This is done before the other norms are calculated to ensure that the brief utterances are being properly annotated.)
  - 32-40% of utterances which have X words or less are considered brief (this has either been three or four words or less, depending on the experiment).
Utterances with politeness modifiers such as “please” and “thank you” are considered polite (and not brief).
  - Some examples:
    - “Move there.”
    - “Hand me the blue.”
    - “I got it.”
  - Polite utterances are marked by indirect language. Some examples:
“I think you should just move to that city.”
“Maybe you could hand me that card.”
“That should work, if it’s ok with you.”
“Sorry, you’re right, thanks.”
Direct utterances are marked by direct language: Some examples:
“Go there and pick that up.”
“I put that card in the discard pile.”
“Move to the nearest city on the board.”
Contextual Factors (Potential for harm: yes/no, Interlocutor authority: yes/no, Time pressure: yes/no)
Potential for harm is determined by:
The event of an epidemic card being drawn
The event of an outbreak
The state of the board (if the outbreak/infection markers are more ¾ of the way to the end)
Interlocutor authority is determined by:
Who’s turn it is (when a participant is speaking and it is not their turn)
Time pressure is determined by:
When the timer is on (90 seconds per turn in the TP condition)

Updating the Model with Data
Once the annotations are complete, the txt files are converted to .arff format (compatible with Weka) in txt_to_arff.py. A data point is created for each utterance.
This program also checks to see if there are any typos in the formatting of the script, and prints the number of the line with the error (so it can be properly fixed).
All re-formatted data is located in data/experiment_data.
The data is split 80/20 (training/testing) from split_data.py.
Inputs: experiment data from data/experiment_data.
Outputs: random data points of testing/training data.
Training data output locations: data/training_data.
Testing data output locations: data/testing_data.
split_data.py also has the ability to split the data 50/50 for 5 x 2 cross-validation.
model_training_and_eval.py updates the model with the aggregation of all the experiment training data and performs various methods of model evaluation (using the calculation of the model’s CPT - conditional probability table).
The final training data (aggregated training data from each experiment) that is used for the model is located in: data/arff_data/hri_008_training_data.arff.
That data is used to update the model in Weka.
The final testing data that is used to evaluate the model is located at: data/arff_data/hri_008_testing_data.arff.
The xml for the model (used in Weka) is located at model/lockshin_bayesnet.xml.
In addition, if one is interested in seeing the various CPTs over each experiment (or each participant in each experiment), the model can be evaluated by calculating the probability distribution tables (given the data) in experimental_stats.py (note that this program takes in the original txt file as input, not the .arff file).
experimental_stats.py calculates:
Per-participant CPTs
Per-experiment CPTs
CPTs using all the data (from all the experiments, similar to model_training_and_eval.py)
