
import pylab as pl
import numpy as np
import pandas as pd
import StringIO
import pydot

from sklearn.tree import DecisionTreeClassifier 
from sklearn import  (metrics, cross_validation, linear_model, preprocessing)
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import grid_search

from pandasql import sqldf
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier, RandomForestRegressor
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

def create_test_submission(filename, prediction):
    content = ['Id,Action']
    for i, p in enumerate(prediction):
        content.append('%i,%0.9f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'

def main(train_file='train.csv', test_file='test.csv', output_file='predict.csv'):
    print "Loading data..."
    
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    y = np.array(train_data[["ACTION"]])
    #X = np.array(train_data.ix[:,1:-1])     # Ignores ACTION, ROLE_CODE
    X = np.array(train_data[["RESOURCE","MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_DEPTNAME", "ROLE_CODE"]])
    X_test = np.array(test_data[["RESOURCE","MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_FAMILY_DESC", "ROLE_FAMILY","ROLE_DEPTNAME", "ROLE_CODE"]]) # Ignores ID, ROLE_CODE
 
    SEED = 4
    #clf = DecisionTreeClassifier(criterion="entropy").fit(X,y)
    
    
    
    clf = RandomForestRegressor(n_estimators=300, min_samples_split=15, min_density=0.1,compute_importances=True).fit(X,y)

    print clf.feature_importances_
    #Try feature selection
    
    mean_auc = 0.0
    n = 10
    for i in range(n):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=.10, random_state=i*SEED)

        # if you want to perform feature selection / hyperparameter
        # optimization, this is where you want to do it
        
        # train model and make predictions
        clf.fit(X_train, y_train) 
        preds = clf.predict(X_cv)

        # compute AUC metric for this CV fold
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        mean_auc += roc_auc
    
    print "Mean AUC: %f" % (mean_auc/n)
    predictions = clf.predict_(X_test)
    #print predictions
    
    #print 'Writing predictions to %s...' % (output_file)
    create_test_submission(output_file, predictions)

    return 0


if __name__=='__main__':
    args = { 'train_file':  'train.csv',
             'test_file':   'test.csv',
             'output_file': 'predict_randomforestregressor.csv' }
    model = main(**args)    

