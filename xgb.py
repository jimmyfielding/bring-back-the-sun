import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
import xgboost as xgb

'''
Read in all required csv files as panda DataFrames
Includes:
    labeled training data
    unlabeled testing data
    more training data with missing values
    confidence annotations of labels in labelled data
'''
training_df = pd.read_csv("training.csv", index_col=0)
testing_df = pd.read_csv("testing.csv", index_col=0)
additional_training_df = pd.read_csv("additional_training.csv", index_col=0)
training_predictions_df = pd.read_csv("training_predictions.csv", index_col=0)
additional_predictions_df = pd.read_csv("additional_predictions.csv", index_col=0)
annotation_confidence_df = pd.read_csv("annotation_confidence.csv", index_col=0)

# Concatenate all training data and training labels into single DataFrames
whole_data_df = pd.concat([training_df, additional_training_df], sort=False)
whole_data_predictions_df = pd.concat([training_predictions_df, additional_predictions_df], sort=False)



confidence = annotation_confidence_df.values
labels = whole_data_predictions_df.values

# Add label and confidence columns to the DataFrame that contains all training data
whole_data_df['labels'] = labels
whole_data_df['confidence'] = confidence

# Filter whole training data DataFrame into two new Dataframes grouped by label
mask = whole_data_df['labels'] == 1
ones_data_df = whole_data_df[mask]
zeros_data_df = whole_data_df[~mask]

# Replace all NaN values with the NaN mean of the feature
non_nan_ones_data_df = ones_data_df.fillna(np.nanmean(ones_data_df))
non_nan_zeros_data_df = zeros_data_df.fillna(np.nanmean(zeros_data_df))

# Reconstruct whole training data DataFrame
frames = [non_nan_ones_data_df, non_nan_zeros_data_df]
whole_data_df = pd.concat(frames)

# Reverse class imbalance; '1' now the minority class and '0' the majority class, to reflect the testing data
# This is performed by dropping all data with a label '1' that also has a confidence value of '0.66' i.e. partially confident in label
indexNames = whole_data_df[(whole_data_df['labels'] == 1) & (whole_data_df['confidence'] == 0.66)].index
whole_data_df.drop(indexNames, inplace=True)

# Trim predictions and annotations to reflex changes in whole training data DataFrame
trimmed_data_predictions = whole_data_df['labels']
trimmed_annotation_confidence = whole_data_df['confidence']

# Drop unnecessary columns now grouping has been performed
whole_data_df.drop(['labels'], axis=1, inplace=True)
whole_data_df.drop(['confidence'], axis=1, inplace=True)

'''
# Perform standardisation on both the training and testing data
whole_data_df[whole_data_df.columns] = Normalizer().fit_transform(whole_data_df[whole_data_df.columns])
testing_df[testing_df.columns] = Normalizer().fit_transform(testing_df[testing_df.columns])
'''

# Create training, labels, testing, confidence numpy arrays ready for model training
X = whole_data_df.values
y = trimmed_data_predictions.values
Z = testing_df.values
c = trimmed_annotation_confidence.values

# Create new RandomForestClassifier
rdfclf = RandomForestClassifier(max_features='auto', max_depth=1000, max_leaf_nodes=10000, n_estimators=100,
                                random_state=123)
# Fit the classifier using the confidence values as sample_weights
rdfclf.fit(X, y, c)

# Input of hyper-parameters for the XGBClassifier
params = {'max_depth': 6, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic', 'scale_pos_weight': 0.67,
          'subsample': 0.9, 'colsample_bytree': 0.7, 'min_child_weight': 9,'interaction_constraints': [[2299, 2432]]}
params['nthread'] = 4
params['eval_metric'] = 'error'
num_trees = 1000

# Train Gradient Booster Classifier
gbm = xgb.train(params, xgb.DMatrix(
    X, y), num_trees)

# Prediction labels using the mean prediction from the two classifiers, enforcing the threshold on the minority class
predictions = (gbm.predict(xgb.DMatrix(Z))> 0.5).astype(int)

# Save results to a csv ready for submission
data = pd.DataFrame({'prediction': predictions})
data.index += 1
data.index.name = 'ID'
submission = data.to_csv('submission.csv')


zerocount = 0
onecount = 1

for i in range(len(predictions)):
    if predictions[i] == 0:
        zerocount += 1
    else:
        onecount += 1

print(onecount)
print(zerocount)

'''
GridSearch Example From Hyper-Parameter Optimisation of the XGB model within this model

parameters = {'nthread':[4], 
              'objective':['binary:logistic'],
              'learning_rate': [1.0, 0.05],
              'max_depth': [6, 7, 8],
              'min_child_weight': [11, 12 ,10, 9],
              'silent': [1],
              'subsample': [0.8, 0.7, 0.9],
              'colsample_bytree': [0.7, 0.8, 0.9],
              'n_estimators': [5, 1000],
              'missing':[-999],
              'seed': [123],
              'scale_pos_weight': [0.67],
              'interaction_constraints': [[[2299, 2432]]]}

skf = StratifiedKFold(n_splits=2)

clf = GridSearchCV(xgb_model, parameters, n_jobs=7,
                   cv=skf,
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(X, y)

print(clf.best_params_)
675
2144
'''
