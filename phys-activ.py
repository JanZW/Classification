#data-source: http://archive.ics.uci.edu/ml/datasets/EMG+Physical+Action+Data+Set
from sklearn.ensemble import RandomForestClassifier #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC#https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
import pandas as pd
import glob, os, time
import pickle

def train_rf(X,y,df_time):
    rf=RandomForestClassifier(warm_start=True,oob_score=True,n_jobs=-1)
    rf_clf=make_pipeline(StandardScaler(),rf)
    error_rate=[]
    for n,filename in zip(N,filename_rf):
        rf.set_params(n_estimators=n)
        rf_clf.fit(X_train,y_train)
        error_rate.append((n,1-rf.oob_score_))
        print('fitted rf modell successfully:',time.time()-df_time)
        pickle.dump(rf_clf,open(filename,'wb'))
        print('done with ',n)
    xs, ys = zip(*error_rate)
    plt.plot(xs, ys)
    plt.show()
    return None
def train_svm(X,y,df_time):
    svm_clf=make_pipeline(StandardScaler(),SVC(kernel='rbf',shrinking=False,cache_size=2000))
    svm_clf.fit(X_train,y_train)
    print('fitted svm modell successfully:',(time.time()-df_time)/60,'minutes')
    pickle.dump(svm_clf, open(filename_svm, 'wb'))
    return None

def test(X,y,bool_rf):
    if not bool_rf:
        model=pickle.load(open(filename_svm, 'rb'))
    else:
        n=int(input('how many trees? (10,50,100,500)'))
        file_rf=filename_rf[N.index(n)]
        model=pickle.load(open(file_rf,'rb'))
    y_pred=model.predict(X)
    metrics.plot_roc_curve(model,X,y)
    roc_auc=metrics.roc_auc_score(y,y_pred)
    accuracy=metrics.accuracy_score(y,y_pred)
    sensitivity_specificity=metrics.classification_report(y,y_pred)
    print('roc_auc:',roc_auc)
    print('Accuracy:',accuracy)
    print('sensitivity_specificity:\n',sensitivity_specificity)
    plt.show()
    return None

def load_data():
    #Load Data into pd.DataFrame and shuffle
    path1_aggressive=r'EMG Pysical Action Data Set/sub1/aggressive/txt'
    path1_normal=r'EMG Pysical Action Data Set/sub1/normal/txt'
    path2_aggressive=r'EMG Pysical Action Data Set/sub2/aggressive/txt'
    path2_normal=r'EMG Pysical Action Data Set/sub2/normal/txt'
    path3_aggressive=r'EMG Pysical Action Data Set/sub3/aggressive/txt'
    path3_normal=r'EMG Pysical Action Data Set/sub3/normal/txt'
    paths_normal=[path1_normal,path2_normal,path3_normal]
    paths_aggressive=[path1_aggressive,path2_aggressive,path3_aggressive]
    df=pd.DataFrame({})
    for path in paths_normal:
        all_files=glob.iglob(os.path.join(path,'*.txt'))
        df=pd.concat((df,pd.concat((pd.read_csv(f,sep='\t',header=None) for f in all_files), ignore_index=True)),ignore_index=True)
    df=df.dropna()
    normal=df.shape[0]
    for path in paths_aggressive:
        all_files=glob.iglob(os.path.join(path,'*.txt'))
        df=pd.concat((df,pd.concat((pd.read_csv(f,sep='\t',header=None) for f in all_files), ignore_index=True)),ignore_index=True)
    df=df.dropna()
    datapoints=df.shape[0]
    labels=np.array([0]*normal+[1]*(datapoints-normal))
    
    #shuffle data and labels
    X,y=shuffle(df, labels)
    #split in train and test data
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    dataframe_time=time.time()
    print('created DataFrame successfully:',dataframe_time-start_time,'seconds') #1 second
    return X_train,X_test,y_train,y_test,dataframe_time

if __name__=="__main__":
    start_time=time.time()
    filename_svm='svm-ex2-1.sav'
    filename_rf=['rf-ex2-1-10.sav','rf-ex2-1-50.sav','rf-ex2-1-100.sav','rf-ex2-1-500.sav']
    N=[10,50,100,500]

    bool_train=int(input('Do you want to train a model? (1/0)\n'))
    bool_rf=int(input("Random Forest? (1/0)\n"))

    X_train,X_test,y_train,y_test,df_time=load_data()
    if bool_train:
        if bool_rf:
            train_rf(X_train,y_train,df_time)
        else:
            train_svm(X_train,y_train,df_time)
    else:
        test(X_test,y_test,bool_rf)