from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

training_df = pd.read_csv("training.csv", index_col=0)
testing_df = pd.read_csv("testing.csv", index_col=0)
additional_training_df = pd.read_csv("additional_training.csv", index_col=0)
training_predictions_df = pd.read_csv("training_predictions.csv", index_col=0)
additional_predictions_df = pd.read_csv("additional_predictions.csv", index_col=0)
annotation_confidence_df = pd.read_csv("annotation_confidence.csv", index_col=0)

whole_data_df = pd.concat([training_df, additional_training_df], sort=False)
whole_data_predictions_df = pd.concat([training_predictions_df, additional_predictions_df], sort=False)

whole_data_df[whole_data_df.columns] = StandardScaler().fit_transform(whole_data_df[whole_data_df.columns])
testing_df[testing_df.columns] = StandardScaler().fit_transform(testing_df[testing_df.columns])

confidence = annotation_confidence_df.values
labels = whole_data_predictions_df.values

whole_data_df['labels'] = labels
whole_data_df['confidence'] = confidence

onecount = 0
zerocount = 0

for i in whole_data_df['labels']:
    if i == 0:
        zerocount += 1
    else:
        onecount += 1

print(onecount)
print(zerocount)


mask = whole_data_df['labels'] == 1

ones_data_df = whole_data_df[mask]
zeros_data_df = whole_data_df[~mask]
non_nan_ones_data_df = ones_data_df.fillna(np.nanmean(ones_data_df))
non_nan_zeros_data_df = zeros_data_df.fillna(np.nanmean(zeros_data_df))
frames = [non_nan_ones_data_df, non_nan_zeros_data_df]

whole_data_df = pd.concat(frames)

difference = onecount - zerocount

indexNames = whole_data_df[(whole_data_df['labels'] == 1) & (whole_data_df['confidence'] == 0.66)].index
difference = len(indexNames) - difference

trunc_index_names = indexNames[:-difference]
whole_data_df.drop(indexNames, inplace=True)

onecount = 0
zerocount = 0

for i in whole_data_df['labels']:
    if i == 0:
        zerocount += 1
    else:
        onecount += 1

print(onecount)
print(zerocount)

trimmed_data_predictions = whole_data_df['labels']
trimmed_annotation_confidence = whole_data_df['confidence']

whole_data_df.drop(['labels'], axis=1, inplace=True)
whole_data_df.drop(['confidence'], axis=1, inplace=True)

print(whole_data_df.head)

X = whole_data_df.values
y = trimmed_data_predictions.values
Z = testing_df.values
c = trimmed_annotation_confidence.values

xgb_model = xgb.XGBClassifier()

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have
#much fun of fighting against overfit
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [1.0, 0.05], #so called `eta` value
              'max_depth': [6, 7, 8],
              'min_child_weight': [11, 12 ,10, 9],
              'silent': [1],
              'subsample': [0.8, 0.7, 0.9],
              'colsample_bytree': [0.7, 0.8, 0.9],
              'n_estimators': [5, 1000], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [123],
              'scale_pos_weight': [0.67],
              'interaction_constraints': [[[2299, 2432]]]}

skf = StratifiedKFold(n_splits=2)

clf = GridSearchCV(xgb_model, parameters, n_jobs=6,
                   cv=skf,
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(X, y)

print(clf.best_params_)

'''
#trust your CV!
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
    '''
