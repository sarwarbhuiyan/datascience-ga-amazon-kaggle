
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
from itertools import combinations

from pandasql import sqldf
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

def group_data(data, degree=3, cutoff = 1, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    
    new_data = []
    m,n = data.shape
    for indexes in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indexes]])
    for z in range(len(new_data)):
        counts = dict()
        useful = dict()
        for item in new_data[z]:
            if item in counts:
                counts[item] += 1
                if counts[item] > cutoff:
                    useful[item] = 1
            else:
                counts[item] = 1
        for j in range(len(new_data[z])):
            if not new_data[z][j] in useful:
                new_data[z][j] = 0
    return np.array(new_data).T

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
    all_data = np.vstack((train_data.ix[:,1:-1], test_data.ix[:,1:-1]))

    num_train = np.shape(train_data)[0]
        
    y = np.array(train_data[["ACTION"]])
    #X = np.array(train_data.ix[:,1:-1])     # Ignores ACTION, ROLE_CODE
    X = np.array(train_data[["RESOURCE","MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE"]])
    X_test = np.array(test_data[["RESOURCE","MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_FAMILY_DESC", "ROLE_FAMILY","ROLE_CODE"]]) # Ignores ID, ROLE_CODE
    
    
    X = all_data[:num_train]
    X_test = all_data[num_train:]
    print "Transforming data..."
    dp1 = group_data(all_data, degree=2, cutoff=2) 
    dt1 = group_data(all_data, degree=3, cutoff=2)
    dz1 = group_data(all_data, degree=4, cutoff=2)
    dp2 = group_data(all_data, degree=5, cutoff=2)
    dp3 = group_data(all_data, degree=6, cutoff=2) 
    X_2  = dp1[:num_train]
    X_3  = dt1[:num_train]
    X_4  = dz1[:num_train]
    X_5  = dp2[:num_train]
    X_6  = dp3[:num_train]
 
    X_test = all_data[num_train:]
    X_test_2 = dp1[num_train:]
    X_test_3 = dt1[num_train:]
    X_test_4 = dz1[num_train:]
    X_test_5 = dp2[num_train:]
    X_test_6 = dp3[num_train:]
 
    X = np.hstack((X, X_2, X_3, X_4, X_5, X_6))
    X_test = np.hstack((X_test, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6))
    
   
    SEED = 4
    
    clf = RandomForestClassifier(n_estimators=200, min_samples_split=15, min_density=0.1,compute_importances=True)
  
    clf.fit(X,y)
    #RESOURCE,MGR_ID,ROLE_FAMILY_DESC
    print clf.feature_importances_
    
    mean_auc = 0.0
    n = 10
    for i in range(n):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=.10, random_state=i*SEED)

        # if you want to perform feature selection / hyperparameter
        # optimization, this is where you want to do it
        
        # train model and make predictions
        clf.fit(X_train, y_train) 
        preds = clf.predict_proba(X_cv)[:,1]

        # compute AUC metric for this CV fold
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        mean_auc += roc_auc
    
    print "Mean AUC: %f" % (mean_auc/n)

    predictions = clf.predict_proba(X_test)[:,1]
    #print predictions
    
    #print 'Writing predictions to %s...' % (output_file)
    create_test_submission(output_file, predictions)

    return 0


if __name__=='__main__':
    args = { 'train_file':  'train.csv',
             'test_file':   'test.csv',
             'output_file': 'predict_randomforest2.csv' }
    model = main(**args)    

