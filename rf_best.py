import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

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

# Perform standardisation on both the training and testing data
whole_data_df[whole_data_df.columns] = Normalizer(norm='l2').fit_transform(whole_data_df[whole_data_df.columns])
testing_df[testing_df.columns] = Normalizer(norm='l2').fit_transform(testing_df[testing_df.columns])

# Create training, labels, testing, confidence numpy arrays ready for model training
X = whole_data_df.values
y = trimmed_data_predictions.values
Z = testing_df.values
c = trimmed_annotation_confidence.values

# Create new RandomForestClassifier
rdfclf = RandomForestClassifier(max_features='auto', max_depth=100, max_leaf_nodes=130, n_estimators=100,
                                random_state=123)
# Fit the classifier using the confidence values as sample_weights
rdfclf.fit(X, y, c)

# Make predictions based on probability using the defined threshold on the '1' label
predictions = (rdfclf.predict_proba(Z)[:, 1] > 0.5).astype(int)

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
RandomisedGridSearch Example

n_estimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(100, 100000, num = 100)]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}


def grid_search_wrapper(refit_score='precision_score'):
    
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train, confidence_train)
    

    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=skf, verbose=2,
                                   random_state=123, n_jobs=5, refit=refit_score)
    rf_random.fit(X_train, y_train, confidence_train)

    y_pred = (rf_random.predict_proba(Z)[:, 1] > 0.6).astype(int)

    print(rf_random.best_params_)

    return rf_random


grid_search_clf = grid_search_wrapper(refit_score='precision_score')

'''
