
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def precision_recall(y_true, y_pred):
    
    pr, rc, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc_score = average_precision_score(y_true, y_pred)
    
    #Retrun dataframe so that we can refer the value to find the ideal threshold. 'AUC' columne store value for the training model
    pr_rc_curve_df = pd.DataFrame()    
    pr_rc_curve_df['precision'] = pr
    pr_rc_curve_df['recall'] = rc
    pr_rc_curve_df['thresholds'] = np.insert(thresholds, len(thresholds), np.nan)
    pr_rc_curve_df['auc'] = pr_auc_score 

    #disp = PrecisionRecallDisplay(precision = pr, recall = rc)
    #disp.plot()
    #plt.show()

    #return pr, rc, thresholds, pr_auc_score
    return pr_rc_curve_df
    
def roc(y_true, y_pred):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    
    roc_auc_curve_df = pd.DataFrame()
    
    roc_auc_curve_df['false_positive_rates'] = fpr
    roc_auc_curve_df['true_positive_rates'] = tpr
    roc_auc_curve_df['thresholds'] = thresholds
    roc_auc_curve_df['auc'] = auc_score
    
    #return fpr, tpr, thresholds, ideal_threshold
    return roc_auc_curve_df

def plot_curve(**kwargs):

    """    
    Will us single method to plot PR-RC and ROC Curve of the training data
    """
    num_cols = len(kwargs.items())
    #_, (ax, ax_new) = plt.subplots( nrows = 1, ncols = num_cols , figsize = (10,6))
    fig, axes = plt.subplots(nrows = 1, ncols = num_cols , figsize = (10,6))
    
    if num_cols == 1:
        axes = [axes]

    i = -1
    for key, value in kwargs.items():

        label = key
        df = value
        i += 1

        if label.upper() == 'roc'.upper():
            
            #plt.plot([0, 1], [0, 1], 'y--', )
            
            axes[i].plot(df['false_positive_rates'], df['true_positive_rates'],  color = 'green', label = 'ROC Curve') #, marker = 'o'

            axes[i].tick_params(axis = 'both', labelcolor = 'green')
            axes[i].set_xlabel('False positive rate')
            axes[i].set_ylabel('True positive rate')

            if 'auc' in df.columns:
                label_str = str.format('ROC-AUC: {0}',  round(df.loc[0,'auc'], 3))
                axes[i].text(0.5, 0, label_str, fontsize = 6)

        elif label.upper() == 'pr_rc'.upper():
           
            #ax_new = ax.twinx().twiny()
            axes[i].plot(df['recall'], df['precision'], color = 'red', label = 'Precision - Recall Curve') #, marker = '-'

            axes[i].tick_params(axis = 'both', labelcolor = 'red')
            axes[i].set_xlabel('Recall')
            axes[i].set_ylabel('Precision')

            if 'auc' in df.columns:
                label_str = str.format('PR-RC-AUC: {0}',  round(df.loc[0,'auc'], 3))
                axes[i].text(0.5, 1, label_str, fontsize = 6)


    #plt.title('Curves')
    plt.show()

def transform_logist_label(y, threshold):
    return y >= threshold

def accuracy_score(y_true, y_pred, threshold):
    
    y_pred = transform_logist_label(y_pred, threshold)
    print(f'Balance accuracy score: {balanced_accuracy_score(y_true, y_pred)} when threshold {threshold}')

def report_classification(y_true, y_pred, threshold):
    
    y_pred = transform_logist_label(y_pred, threshold)
    print(f"Classification report with threshold {threshold}")
    print(classification_report(y_true, y_pred))

def plot_confusion_matric(y_true, y_pred, threshold):
    
    y_pred = transform_logist_label(y_pred, threshold)
    conf_matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap = plt.cm.Blues, alpha=0.3)

    for i in range(conf_matrix.shape[0]):

        for j in range(conf_matrix.shape[1]):

            ax.text(x = j, y = i,s = conf_matrix[i, j], va = 'center', ha = 'center', size = 'xx-large')


    plt.ylabel('Actuals', fontsize=18)
    plt.xlabel('Predictions', fontsize=18)
    plt.title(f'Confusion Matrix (threshold: {threshold})', fontsize = 18)
    plt.show()