import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, recall_score, precision_score,
                             matthews_corrcoef)
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
# This is simply an alias for convenience
layers = tf.keras.layers

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

    fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
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
        dataset = DataList (unless you create different list)
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
        dataset = DataList (unless you create different list)
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
    
    print("Model performance for %s set:" %title)
    print("--------------------------------------------------------")
    print("Accuracy  : {:.2f}".format(accuracy*100) + "%")
    print("Precision : {:.2f}".format(precision*100) + "%")
    print("Recall    : {:.2f}".format(recall*100) + "%")
    print("F1-score  : {:.2f}".format(f1*100) + "%")
    print("MCC       : {:.2f}".format(mcc*100) + "%")
    plot_confusion_matrix(labels, predictions, ["beam","reaction"])
    print()
    
def make_nn_plots(history, min_acc = 0.95):
    
    """This function prints performance of a neural network per epoch (model loss & accuracy)
    
    Arguments:
        history: history object obtained when fitting a tensorflow neural network
        min_acc: set min value for accuracy. By default the plot has y_min=0.95
    Returns:
        None
    """
    
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    num_epochs = len(history.history['loss'])

    # Loss **********************
    ax[0].set_title("Model Loss")
    # X-axis
    ax[0].set_xlabel("Epoch")
    ax[0].set_xlim(1,num_epochs)
    ax[0].set_xticks(range(1,num_epochs+1))
    # Y-axis
    ax[0].set_ylabel("Loss")

    # Plotting
    ax[0].plot(range(1,num_epochs+1),history.history['loss'], label='Training')
    ax[0].plot(range(1,num_epochs+1),history.history['val_loss'], label='Validation')  
    ax[0].legend()

    # Accuracy **********************
    ax[1].set_title("Model Accuracy")
    # X-axis
    ax[1].set_xlabel("Epoch")
    ax[1].set_xlim(1,num_epochs)
    ax[1].set_xticks(range(1,num_epochs+1))
    # Y-axis
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(min_acc,1)
    # Plotting
    ax[1].plot(range(1,num_epochs+1),history.history['acc'], label='Training')
    ax[1].plot(range(1,num_epochs+1),history.history['val_acc'], label='Validation')
    ax[1].legend()
    
    
def load_data(hf):
    
    """This function loads the dataset and removes empty events
    
    Arguments:
        hf: file containing dataset
    Returns:
        AllData : list containing features for nonempty events
        Labels: numpy ndarray containing labels for nonempty events
    """
    # Importing the data in a python list: AllDataList 
    AllDataList = []
    for i in range(len(hf.keys())):
        KeyString = "Event_[" + str(i) +"]"
        AllDataList.append(hf[KeyString][:]) # each list element is an event 2d matrix
    
    
    # Assigning lables and cleaning empy events
    
    # List of labels 
    LabelsList = []
    # Selecting Empty Events 
    EmptyDataList = []

    # List of not empty Events
    DataList = []
    
    beam = 0 # non-empty beam events
    reaction = 0 # non-empty reaction events
    
    for i in range(len(AllDataList)):
        if (len(AllDataList[i])>0): # Choosing only not empy events
            DataList.append(AllDataList[i]) 
            
            if (i%2==0): # Beam Event (even) --> label = 0
                LabelsList.append(0)
                beam = beam + 1
            else:        # Reaction Event (odd)     --> label = 1
                LabelsList.append(1)
                reaction = reaction + 1
        else:
            EmptyDataList.append(i)
                
    # Converting List in a Numpy Array: it is faster and easier to handle.
    Labels = np.array(LabelsList)

    print("Dataset contains " + str(len(AllDataList)) + " events")
    print("Data contains %i empty events, of indexes:"  % len(EmptyDataList),  EmptyDataList)
    print("")
    print("Dataset contains " + str(len(DataList)) + " non-empty events:")
    print(str(beam) + " Beam Events, and " + str(reaction) + " Reaction Events\n")
    
    return DataList, Labels




def calc_features(DataList):
    
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
    
    MeanXPerEvent = np.zeros(len(DataList))
    MeanYPerEvent = np.zeros(len(DataList))
    MeanZPerEvent = np.zeros(len(DataList))
    StDevXPerEvent = np.zeros(len(DataList))
    StDevYPerEvent = np.zeros(len(DataList))
    StDevZPerEvent = np.zeros(len(DataList))
    MeanWeightedXPerEvent = np.zeros(len(DataList))
    MeanWeightedYPerEvent = np.zeros(len(DataList))
    MeanWeightedZPerEvent = np.zeros(len(DataList))

    SumAPerEvent = np.zeros(len(DataList))
    PadsPerEvent = np.zeros(len(DataList))
    FracClosePtsPerEvent = np.zeros(len(DataList)) # fraction of points satisfying x^2+y^2<100 in event


    # Computing the features
    for i in range(len(DataList)):
        
        PadsPerEvent[i] = len(DataList[i])
        
        # Calculating mean values, FCP and Total Q
        for j in range(len(DataList[i])):
            MeanXPerEvent[i] = MeanXPerEvent[i] + DataList[i][j][0]
            MeanYPerEvent[i] = MeanYPerEvent[i] + DataList[i][j][1]
            MeanZPerEvent[i] = MeanZPerEvent[i] + DataList[i][j][2] 
            MeanWeightedXPerEvent[i] = MeanWeightedXPerEvent[i] + DataList[i][j][0]*DataList[i][j][4]
            MeanWeightedYPerEvent[i] = MeanWeightedYPerEvent[i] + DataList[i][j][1]*DataList[i][j][4]
            MeanWeightedZPerEvent[i] = MeanWeightedZPerEvent[i] + DataList[i][j][2]*DataList[i][j][4]  
            SumAPerEvent[i] = SumAPerEvent[i] + DataList[i][j][4]
            
            if (DataList[i][j][0]**2 + DataList[i][j][1]**2 < 100):
                FracClosePtsPerEvent[i] = FracClosePtsPerEvent[i] + 1 
        
        MeanXPerEvent[i] = MeanXPerEvent[i]/len(DataList[i])
        MeanYPerEvent[i] = MeanYPerEvent[i]/len(DataList[i])
        MeanZPerEvent[i] = MeanZPerEvent[i]/len(DataList[i])    
        MeanWeightedXPerEvent[i] = MeanWeightedXPerEvent[i]/len(DataList[i])
        MeanWeightedYPerEvent[i] = MeanWeightedYPerEvent[i]/len(DataList[i])
        MeanWeightedZPerEvent[i] = MeanWeightedZPerEvent[i]/len(DataList[i])  
        FracClosePtsPerEvent[i] = FracClosePtsPerEvent[i]/len(DataList[i])
    
        # Second for loop for calculation of standard deviation
        for j in range(len(DataList[i])):
            
            StDevXPerEvent[i] = StDevXPerEvent[i] + (DataList[i][j][0]-MeanXPerEvent[i])**2
            StDevYPerEvent[i] = StDevYPerEvent[i] + (DataList[i][j][1]-MeanYPerEvent[i])**2
            StDevZPerEvent[i] = StDevZPerEvent[i] + (DataList[i][j][2]-MeanZPerEvent[i])**2
            
        StDevXPerEvent[i] = np.sqrt(StDevXPerEvent[i])/(len(DataList[i])-1)
        StDevYPerEvent[i] = np.sqrt(StDevYPerEvent[i])/(len(DataList[i])-1)
        StDevZPerEvent[i] = np.sqrt(StDevZPerEvent[i])/(len(DataList[i])-1)
        
    return (MeanXPerEvent, MeanYPerEvent, MeanZPerEvent, SumAPerEvent, PadsPerEvent,
    MeanWeightedXPerEvent, MeanWeightedYPerEvent, StDevXPerEvent, StDevYPerEvent, StDevZPerEvent,FracClosePtsPerEvent)


def build_pretrained_vgg_model(input_shape, num_classes):
    """Constructs a CNN with a VGG16's convolutional base and two fully-connected hidden layers on top.
    The convolutional base is frozen (the weights can't be updated) and has weights from training on the ImageNet dataset.

    Returns:
    The model.
    """
# This loads the VGG16 model from TensorFlow with ImageNet weights
    vgg_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    
# First we flatten out the features from the VGG16 model
    net = layers.Flatten()(vgg_model.output)

# We create a new fully-connected layer that takes the flattened features as its input
    net = layers.Dense(512, activation=tf.nn.relu)(net)
# And we add one more hidden layer
    net = layers.Dense(512, activation=tf.nn.relu)(net)

# Then we add a final layer which is connected to the previous layer and
# groups our images into one of the three classes
    output = layers.Dense(num_classes, activation=tf.nn.softmax)(net)

# Finally, we create a new model whose input is that of the VGG16 model and whose output
# is the final new layer we just created
    model = tf.keras.Model(inputs=vgg_model.input, outputs=output)
    
# We loop through all layers except the last four and specify that we do not want 
# their weights to be updated during training. Again, the weights of the convolutional
# layers have already been trained for general-purpose feature extraction, and we only
# want to update the fully-connected layers that we just added.
    for layer in model.layers[:-4]:
        layer.trainable = False

    return model


def normalize_image_data(images):
    """ Takes an imported set of images and normalizes values to between
    0 and 1 using min-max scaling across the whole image set.
    """
    img_max = np.amax(images)
    img_min = np.amin(images)
    #Debug 
    print("The max value of images is: ", img_max, " while the minimum is: ", img_min)
    if img_max==0:
        print("Error: File given is made by black images (only zeros)")
    else: 
        if (img_max - img_min) > 0:
            images = (images - img_min) / (img_max - img_min)
        else: 
            images = (images - img_min) / img_max
            print("Error: File given is made by same values images, now it has been normalized to 1")
            
    return images