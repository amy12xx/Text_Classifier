# Text_Classifier
Text_Classifier for classifying free online courses using scikit-learn and nltk

1. /data folder contains training courses data
2. make_data_set_pickle.py converts training data from different course files in /data to single pickle file to be used by Text_classifier
3. Text_classifier.py performs classification of courses into their categories (such as Math, Physics, Economics etc) using Support Vector Machines in scikit-learn.
4. text_processor.py is used for common text processing tasks by text_classifier.
