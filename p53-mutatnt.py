#data source: http://archive.ics.uci.edu/ml/machine-learning-databases/p53/
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
import pickle
from sklearn.model_selection import train_test_split


def load_data():
    print('loading data... (this can take several minutes)')
    start_time=time()
    X=np.loadtxt('p53_old_2010/K8.data',\
        usecols=(k for k in range(5407)),\
        delimiter=',',\
        converters = {k: lambda s:float(s.replace(b'?',b'').strip() or np.nan)\
             for k in range(5407)})
    y=np.loadtxt('p53_old_2010/K8.data',usecols=5408,delimiter=',',dtype=str)
    #Drop rows with nan
    idx=~np.isnan(X).any(axis=1)
    X=X[idx]
    y=y[idx]

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

    print('data loaded in',(time()-start_time)/60,'minutes')#5.58 minutes
    return X_train,X_test,y_train,y_test

def train_rf():
    print('training random forest...')
    rf=RandomForestClassifier(warm_start=True,oob_score=True,n_jobs=-1)
    rf_clf=make_pipeline(StandardScaler(),rf)#scaling not necessary
    error_rate=[]
    N=[100,500,1000,2000,5408]
    filename='rf-ex2-3-'
    for n in N:
        start_time=time()
        rf.set_params(n_estimators=n)
        rf_clf.fit(X_train,y_train)
        print('done with ',n,'in',time()-start_time,'seconds')
        error_rate.append((n,1-rf.oob_score_))
        pickle.dump(rf_clf,open(filename+str(n)+'.sav','wb'))
    xs, ys = zip(*error_rate)
    plt.plot(xs, ys)
    plt.show()
    return None

def train_svm():
    print('training svm...')
    start_time=time()
    svm_clf=make_pipeline(StandardScaler(),LinearSVC(max_iter=40000)) 
    svm_clf.fit(X_train,y_train)
    print('fit completed in',(time()-start_time)/60,'minutes')#17.37 minutes
    filename='trained-model-2-3.sav'
    pickle.dump(svm_clf, open(filename, 'wb'))
    return None

def test(X,y,bool_rf):
    N=[100,500,1000,2000,5408]
    if not bool_rf:
        filename_svm='trained-model-2-3.sav'
        model=pickle.load(open(filename_svm, 'rb'))
    else:
        n=int(input('how many trees?'+str(N)))
        file_rf='rf-ex2-3-'+str(n)+'.sav'
        model=pickle.load(open(file_rf,'rb'))
    y_pred=model.predict(X)
    metrics.plot_roc_curve(model,X,y)
    accuracy=metrics.accuracy_score(y,y_pred)
    sensitivity_specificity=metrics.classification_report(y,y_pred)
    print('Accuracy:',accuracy)
    print('sensitivity_specificity:',sensitivity_specificity)
    return None

    

if __name__=='__main__':
    X_train,X_test,y_train,y_test=load_data()
    train_bool=int(input('do you want to train a model? (0/1)\n'))
    if train_bool:
        rf_bool=int(input('do you want to train a random forest? (0/1)\n'))
        if rf_bool:
            train_rf()
        else:
            train_svm()
    else:
        rf_bool=int(input('do you want to load a random forest?(1/0)'))
        test(X_test,y_test,rf_bool)
    plt.show()
