#data source: http://archive.ics.uci.edu/ml/datasets/Breast+Tissue
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time

def load_data():
    start_time=time()
    df=pd.read_excel('BreastTissue.xls',sheet_name='Data',usecols='C:K')
    labels=np.transpose(pd.read_excel('BreastTissue.xls',sheet_name='Data',usecols='B').to_numpy())[0]
    df_time=time()
    print('created DataFrame successfully:',df_time-start_time)#0.037 seconds
    return df, labels

def preprocess(X,y):
    #merge fad, mas and gla to class m
    M={"fad","mas","gla"}
    for k in range(len(y)):
        if y[k] in M:
            y[k]='m'
    lb=LabelBinarizer()
    y_sparse=lb.fit_transform(y)
    classes=['m','adi','car','con']
    s=lb.transform(classes)
    print(classes)
    print(s)
    X_train,X_test,y_train,y_test=train_test_split(X,y_sparse,test_size=0.5)
    return X_train,X_test,y_train,y_test

def train_svm(df,labels,kernel):
    start_time=time()
    if kernel=='linear':
        svm_clf=make_pipeline(StandardScaler(),OneVsRestClassifier(LinearSVC(),n_jobs=-1))
    else:
        svm_clf=make_pipeline(StandardScaler(),OneVsRestClassifier(SVC(kernel=kernel))) 
    svm_clf.fit(df,labels)
    print('fitted modell successfully:',time()-start_time,'seconds')#0.008 seconds
    return svm_clf

def train_rf(df,labels):
    rf_clf=RandomForestClassifier(oob_score=True,n_jobs=-1,warm_start=True)
    rf_clf.fit(X,y)
    return rf_clf

def test(X,y,model,rf_bool,kernel=''):
    """
    in large parts taken from 
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    if rf_bool:
        y_pred=model.predict(X)
        y_pred=LabelBinarizer().fit_transform(y_pred)
        y_score=y_pred
    else:
        y_score = model.decision_function(X)
        y_pred=model.predict(X)
    n_classes=4
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y[:,i], y_score[:,i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if rf_bool:
        plt.title('Receiver operating characteristic (RF)')
    else:
        plt.title('Receiver operating charactersitic (SVM - '+kernel+')')
    plt.legend(loc="lower right")
    
    accuracy=metrics.accuracy_score(y,y_pred)
    sensitivity_specificity=metrics.classification_report(y,y_pred)
    print('Accuracy:',accuracy)
    print('sensitivity_specificity:\n',sensitivity_specificity)

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = ['aqua', 'darkorange', 'cornflowerblue','black']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if rf_bool:
        plt.title('Multiclass Receiver operating characteristic (RF)')
    else:
        plt.title('Multiclass Receiver operating characteristic (SVM - '+kernel+')')
    plt.legend(loc="lower right")
    

if __name__=='__main__':
    X,y=load_data()
    X_train,X_test,y_train,y_test=preprocess(X,y)
    print('-------SVM-------')
    kernels=['linear','rbf','sigmoid']
    for kernel in kernels:
        print('\n------------'+kernel+'-----------------\n')
        clf=train_svm(X_train,y_train,kernel)
        test(X_test,y_test,clf,False,kernel)
    print('------RF------')
    clf=train_rf(X_train,y_train)
    test(X_test,y_test,clf,True)
    plt.show()
    
