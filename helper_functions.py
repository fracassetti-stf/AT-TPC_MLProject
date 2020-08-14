import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, recall_score, precision_score,
                             matthews_corrcoef)
from mpl_toolkits.mplot3d import Axes3D

def plot_confusion_matrix(y_true,
                          y_pred,
                          classes,
                          title=None,
                          cmap=plt.cm.Blues):
    
    """This function prints and plots the confusion matrix.
    
    Adapted from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    Arguments:
        y_true: Real class labels.
        y_pred: Predicted class labels.
        classes: List of class names.
        title: Title for the plot.
        cmap: Colormap to be used.
    
    Returns:
        None.
    """
    if not title:
        title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    plt.show()
    
def plot_3d_event(dataset,
                  labels,
                  idx):
    
    """This function plots a single event in a 3d plot
    with x,y,z for each pad fired
    
    Arguments:
        dataset = AllData (unless you create different list)
        labels = Labels (used for setting beam or reaction in title)
        idx = index of event in dataset
    
    Returns:
        None
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    xvalues = np.zeros(len(dataset[idx]))
    yvalues = np.zeros(len(dataset[idx]))
    zvalues = np.zeros(len(dataset[idx]))
    
    for i in range(len(dataset[idx])):
        xvalues[i] = dataset[idx][i][0]
        yvalues[i] = dataset[idx][i][1]
        zvalues[i] = dataset[idx][i][2]
    
    
    ax.scatter(xvalues, yvalues, zvalues, marker='o')
    if (labels[idx]==0):
        ax.set_title('Beam Event #' + str(idx), pad = 15, fontsize = 14)
    else:
        ax.set_title('Reaction Event #' + str(idx), pad = 15, fontsize = 14)
    
    ax.set_xlabel('X[mm]')
    ax.set_ylabel('Y[mm]')
    ax.set_zlabel('Z[mm]')
    plt.show()
    
def print_model_performance(labels, 
                            predictions,
                            title = "INPUT SET TYPE"):
    
    """This function prints performance statistics of a model: confusion matrix, precision,
    f1-score and mathews correlation coefficient.
    
    Arguments:
        dataset = AllData (unless you create different list)
        labels = Labels (used for setting beam or reaction in title)
        idx = index of event in dataset
    
    Returns:
        None
    """
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    confmat = confusion_matrix(labels, predictions)
    f1 = f1_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    
    print("Model performance for %s set:"%title)
    print("--------------------------------------------------------\n")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("MCC:", mcc)
    plot_confusion_matrix(labels, predictions, ["beam","reaction"])
    print()
    
def make_nn_plots(history):
    
    """This function prints performance of a neural network per epoch (model loss & accuracy)
    
    Arguments:
        history: history object obtained when fitting a tensorflow neural network
    Returns:
        None
    """
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    num_epochs = len(history.history['loss'])
    ax[0].plot(history.history['loss'], label='training')
    ax[0].plot(history.history['val_loss'], label='validation')
    ax[0].set_title("Model loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_xticks(np.arange(num_epochs))
    ax[0].legend()

    ax[1].plot(history.history['accuracy'], label='training')
    ax[1].plot(history.history['val_accuracy'], label='validation')
    ax[1].set_title("Model accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xticks(np.arange(num_epochs))
    ax[1].legend()
    
def load_data(file):
    
    """This function loads the dataset and removes empty events
    
    Arguments:
        file: file containing dataset
    Returns:
        AllData : list containing features for nonempty events
        Labels: numpy ndarray containing labels for nonempty events
    """
    
    AllDataList = []
    for i in range(len(file.keys())):
        KeyString = "Event_[" + str(i) +"]"
        AllDataList.append(file[KeyString][:])

    print("Dataset contains " + str(len(AllDataList)) + " events")
    
    
    LabelsList = []
    EmptyDataList = []
    AllData = []
    for i in range(len(AllDataList)):
        if (len(AllDataList[i])>0):
            AllData.append(AllDataList[i])
            if (i%2==0):
                LabelsList.append(1)
            else:
                LabelsList.append(0)
        else:
            EmptyDataList.append(i)




    Labels = np.array(LabelsList)

    print("Dataset contains " + str(len(AllData)) + " non-empty events")
    print("Data contains %i empty events, of indexes:"  % len(EmptyDataList),  EmptyDataList)
    return AllData, Labels