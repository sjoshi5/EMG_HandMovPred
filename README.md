# EMG_HandMovPred
A hand movement prediction system using EMG and EEG signals

A dataset of EMG signals has been considered for prediction.
7 types of movements were considered and the data was labelled accordingly to be
0 - unmarked data,
1 - hand at rest, 
2 - hand clenched in a fist, 
3 - wrist flexion,
4 – wrist extension,
5 – radial deviations,
6 - ulnar deviations,
7 - extended palm

The data is in the form of 9 columns with 1 for time and 8 for the 8 channels of EMG signal. 
The tenth column is for specifying the movement.

The code merges datasets of different subjects as per the choice of user.
Merged data is the separated in training and testing sets and the accuracy is checked for KNN, SVM and Naive Bayes.

KNN displayed highest accuracy on the raw data at 96.7%


