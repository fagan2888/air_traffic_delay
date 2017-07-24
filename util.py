
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


# Binary evaluation metrics
def eval_performance(y_true, y_pred, plot=False):

    # scores F1, precision, and recall
    scores = {
        'prec ': metrics.precision_score(y_true, y_pred),
        'recall ': metrics.recall_score(y_true, y_pred),
        'f1': metrics.f1_score(y_true, y_pred)
    }

    for k, v in sorted(scores.items()):
        print('{}\t: {:.3f}'.format(k, v))

    # precision/recall plot
    prec, rec, threshold = metrics.precision_recall_curve(y_true, y_pred)

    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="upper left")
        plt.ylim([0, 2])

    if plot:
        plot_precision_recall_vs_threshold(prec, rec, threshold)
        plt.show()

    # ROC plot
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)

    def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.title('ROC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    if plot:
        plot_roc_curve(fpr, tpr)
        plt.show()


# Keras callbacks plotting
def plot_me(log):
    for k, v in log.history.items():
        if 'acc' in k:
            plt.plot(v, label=k)
            plt.legend()
    plt.title('accuracy');plt.show()
    for k, v in log.history.items():
        if 'loss' in k:
            plt.plot(v, label=k)
            plt.legend()
    plt.title('loss');plt.show()


# CatBoost features importances
def features_report(data_cols, importance, top_n=13):
    d = dict(zip(data_cols, importance))
    s = sorted(d.items(), key=lambda x: x[1]) 
    t = s[::-1]                               
    return t[:top_n]


def visualize_geo(df, long_col='longitude_deg', lat_col='latitude_deg', size=None):
    """visualize the geographical data as a scatter plot"""
    df.plot.scatter(x=long_col, y=lat_col, marker='.', figsize=size)
    plt.gca().set_frame_on(False)
    plt.axis('off')
    plt.show()


