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
        ax.set_title('Beam Event', pad = 15, fontsize = 14)
    else:
        ax.set_title('Reaction Event', pad = 15, fontsize = 14)
    
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
    AllData = []
    for i in range(len(AllDataList)):
        if (len(AllDataList[i])>0):
            AllData.append(AllDataList[i])
            if (i%2==0):
                LabelsList.append(1)
            else:
                LabelsList.append(0)



    Labels = np.array(LabelsList)

    print("Dataset contains " + str(len(AllData)) + " non-empty events")
    
    return AllData, Labels

def best_3cl_km(clust3, x_train, x_val, labels_train):
    
    """This function loads the dataset and removes empty events
    Instead of x_val you can also pass x_test if needed
    Only x_train and labels_train are used in finding the best mapping
    
    Arguments:
        clust3 : fitted 3-cluster kmeans method
        x_train : training features
        x_val : validation features
        labels_train : training labels
    Returns:
        KM3_pred_train : 3-cluster kmeans predictions for training set
        KM3_pred_val : numpy ndarray containing labels for nonempty events
    """
    
    KM3_pred_train = clust3.predict(x_train)
    KM3_pred_val = clust3.predict(x_val)

    #Now we need to find out which cluster is which type of event
    #We select the one that gives best accuracy on training set
    #for this we loop over all combinations without trivial ones (all beam or all reaction)
    #we denote accuracy_train_010 if cluster 0->0, 1->1 and 2->0

    KM3_pred_train_001 = np.zeros(len(KM3_pred_train))
    KM3_pred_train_010 = np.zeros(len(KM3_pred_train))
    KM3_pred_train_011 = np.zeros(len(KM3_pred_train))
    KM3_pred_train_100 = np.zeros(len(KM3_pred_train))
    KM3_pred_train_101 = np.zeros(len(KM3_pred_train))
    KM3_pred_train_110 = np.zeros(len(KM3_pred_train))
    KM3_pred_val_001 = np.zeros(len(KM3_pred_val))
    KM3_pred_val_010 = np.zeros(len(KM3_pred_val))
    KM3_pred_val_011 = np.zeros(len(KM3_pred_val))
    KM3_pred_val_100 = np.zeros(len(KM3_pred_val))
    KM3_pred_val_101 = np.zeros(len(KM3_pred_val))
    KM3_pred_val_110 = np.zeros(len(KM3_pred_val))

    for i in range(len(KM3_pred_train)):
        if (KM3_pred_train[i]==0):
            KM3_pred_train_001[i] = 0
            KM3_pred_train_010[i] = 0
            KM3_pred_train_011[i] = 0
            KM3_pred_train_100[i] = 1
            KM3_pred_train_101[i] = 1
            KM3_pred_train_110[i] = 1
        if (KM3_pred_train[i]==1):
            KM3_pred_train_001[i] = 0
            KM3_pred_train_010[i] = 1
            KM3_pred_train_011[i] = 1
            KM3_pred_train_100[i] = 0
            KM3_pred_train_101[i] = 0
            KM3_pred_train_110[i] = 1
        if (KM3_pred_train[i]==2):
            KM3_pred_train_001[i] = 1
            KM3_pred_train_010[i] = 0
            KM3_pred_train_011[i] = 1
            KM3_pred_train_100[i] = 0
            KM3_pred_train_101[i] = 1
            KM3_pred_train_110[i] = 0

    for i in range(len(KM3_pred_val)):
        if (KM3_pred_val[i]==0):
            KM3_pred_val_001[i] = 0
            KM3_pred_val_010[i] = 0
            KM3_pred_val_011[i] = 0
            KM3_pred_val_100[i] = 1
            KM3_pred_val_101[i] = 1
            KM3_pred_val_110[i] = 1
        if (KM3_pred_val[i]==1):
            KM3_pred_val_001[i] = 0
            KM3_pred_val_010[i] = 1
            KM3_pred_val_011[i] = 1
            KM3_pred_val_100[i] = 0
            KM3_pred_val_101[i] = 0
            KM3_pred_val_110[i] = 1
        if (KM3_pred_val[i]==2):
            KM3_pred_val_001[i] = 1
            KM3_pred_val_010[i] = 0
            KM3_pred_val_011[i] = 1
            KM3_pred_val_100[i] = 0
            KM3_pred_val_101[i] = 1
            KM3_pred_val_110[i] = 0




    accuracy_train_001 = accuracy_score(labels_train, KM3_pred_train_001)
    accuracy_train_010 = accuracy_score(labels_train, KM3_pred_train_010)
    accuracy_train_011 = accuracy_score(labels_train, KM3_pred_train_011)
    accuracy_train_100 = accuracy_score(labels_train, KM3_pred_train_100)
    accuracy_train_101 = accuracy_score(labels_train, KM3_pred_train_101)
    accuracy_train_110 = accuracy_score(labels_train, KM3_pred_train_110)

    KM3_tr_acc_list = [accuracy_train_001, accuracy_train_010, accuracy_train_011, accuracy_train_100, 
                       accuracy_train_101, accuracy_train_110]
    #Uncomment to get the accuracies of the different mappings
    #print(KM3_tr_acc_list)

    #Finds best accuracy model
    max_accuracy_KM3_train = max(KM3_tr_acc_list)
    max_index_KM3 = KM3_tr_acc_list.index(max_accuracy_KM3_train)

    if (max_index_KM3==0):
        KM3_pred_train = KM3_pred_train_001
        KM3_pred_val = KM3_pred_val_001
    elif (max_index_KM3==1):
        KM3_pred_train = KM3_pred_train_010
        KM3_pred_val = KM3_pred_val_010
    elif (max_index_KM3==2):
        KM3_pred_train = KM3_pred_train_011
        KM3_pred_val = KM3_pred_val_011
    elif (max_index_KM3==3):
        KM3_pred_train = KM3_pred_train_100
        KM3_pred_val = KM3_pred_val_100
    elif (max_index_KM3==4):
        KM3_pred_train = KM3_pred_train_101
        KM3_pred_val = KM3_pred_val_101
    elif (max_index_KM3==5):
        KM3_pred_train = KM3_pred_train_110
        KM3_pred_val = KM3_pred_val_110
    
    return KM3_pred_train, KM3_pred_val

def calc_features(AllData):
    
    """This function calculates the features for all events
    
    Arguments:
        AllData : array containing all event data
    Returns:
        MeanXPerEvent : mean x per event
        MeanYPerEvent : mean y per event
        MeanZPerEvent : mean z per event
        SumAPerEvent : sum of charges deposited per event
        PadsPerEvent : nr of pads fired per event
        MeanWeightedXPerEvent : weighted mean x per event
        MeanWeightedYPerEvent : weighted mean y per event
        StDevXPerEvent : standard deviation of x per event
        StDevYPerEvent : standard deviation of y per event
        StDevZPerEvent : standard deviation of z per event
        FracClosePtsPerEvent : fraction of points close to z-axis per event (satisfying x^2+y^2<100)
    """
    
    
    MeanXPerEvent = np.zeros(len(AllData))
    MeanYPerEvent = np.zeros(len(AllData))
    MeanZPerEvent = np.zeros(len(AllData))
    SumAPerEvent = np.zeros(len(AllData))
    PadsPerEvent = np.zeros(len(AllData))
    MeanWeightedXPerEvent = np.zeros(len(AllData))
    MeanWeightedYPerEvent = np.zeros(len(AllData))
    StDevXPerEvent = np.zeros(len(AllData))
    StDevYPerEvent = np.zeros(len(AllData))
    StDevZPerEvent = np.zeros(len(AllData))
    FracClosePtsPerEvent = np.zeros(len(AllData))
    
    for i in range(len(AllData)):

        for j in range(len(AllData[i])):
            MeanXPerEvent[i] = MeanXPerEvent[i] + AllData[i][j][0]
            MeanYPerEvent[i] = MeanYPerEvent[i] + AllData[i][j][1]
            MeanZPerEvent[i] = MeanZPerEvent[i] + AllData[i][j][2]
            SumAPerEvent[i] = SumAPerEvent[i] + AllData[i][j][4]
            MeanWeightedXPerEvent[i] = MeanWeightedXPerEvent[i] + AllData[i][j][0]*AllData[i][j][4]
            MeanWeightedYPerEvent[i] = MeanWeightedYPerEvent[i] + AllData[i][j][1]*AllData[i][j][4]
            if (AllData[i][j][0]**2 + AllData[i][j][1]**2 < 100):
                FracClosePtsPerEvent[i] = FracClosePtsPerEvent[i] + 1 


        MeanXPerEvent[i] = MeanXPerEvent[i]/len(AllData[i])
        MeanYPerEvent[i] = MeanYPerEvent[i]/len(AllData[i])
        MeanZPerEvent[i] = MeanZPerEvent[i]/len(AllData[i])
        MeanWeightedXPerEvent[i] = MeanWeightedXPerEvent[i]/len(AllData[i])
        MeanWeightedYPerEvent[i] = MeanWeightedYPerEvent[i]/len(AllData[i])
        FracClosePtsPerEvent[i] = FracClosePtsPerEvent[i]/len(AllData[i])

        #second for loop for calculation of standard deviation
        for j in range(len(AllData[i])):
            StDevXPerEvent[i] = StDevXPerEvent[i] + (AllData[i][j][0]-MeanXPerEvent[i])**2
            StDevYPerEvent[i] = StDevYPerEvent[i] + (AllData[i][j][1]-MeanYPerEvent[i])**2
            StDevZPerEvent[i] = StDevZPerEvent[i] + (AllData[i][j][2]-MeanZPerEvent[i])**2

        StDevXPerEvent[i] = np.sqrt(StDevXPerEvent[i])/(len(AllData[i])-1)
        StDevYPerEvent[i] = np.sqrt(StDevYPerEvent[i])/(len(AllData[i])-1)
        StDevZPerEvent[i] = np.sqrt(StDevZPerEvent[i])/(len(AllData[i])-1)


        PadsPerEvent[i] = len(AllData[i])
        
    return (MeanXPerEvent, MeanYPerEvent, MeanZPerEvent, SumAPerEvent, PadsPerEvent,
    MeanWeightedXPerEvent, MeanWeightedYPerEvent, StDevXPerEvent, StDevYPerEvent, StDevZPerEvent,FracClosePtsPerEvent)
