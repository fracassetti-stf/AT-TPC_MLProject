import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, recall_score, precision_score,
                             matthews_corrcoef)

# This is simply an alias for convenience
layers = tf.keras.layers





###########################################################################################################################
###############     Data Import Section                                                                     ###############     
###############     Functions dedicated at the first part of the report (i.e. before any ML algorithms):    ###############
###############     data import, data pre-processing, and data-visualization.                               ###############
###########################################################################################################################

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


###########################################################################################################################
###############     Model Performances Section                                                              ###############     
###############     Functions dedicated at the visualization/claculations of the model parformances         ###############
###############                                                                                             ###############
###########################################################################################################################


def make_nn_plots(history, min_acc = 0.95):
    
    """This function prints performance of a neural network per epoch (model loss & accuracy)
    
    Arguments:
        history: history object obtained when fitting a tensorflow neural network
        min_acc: set min axis value for accuracy. By default the plot has y_min=0.95
    Returns:
        None
    """
    
    
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    num_epochs = len(history.history['loss'])
    
    # avoid plotting every epochs ticks (e.g. in autoencoders num_epochs is definitely too  high)
    max_x_ticks = 10 
    if num_epochs > max_x_ticks: 
        x_step = math.floor(num_epochs / max_x_ticks)
    else:
        x_step = 1
    # Loss **********************
    ax[0].set_title("Model Loss")
    # X-axis
    ax[0].set_xlabel("Epoch")
    ax[0].set_xlim(1,num_epochs)
    ax[0].set_xticks(range(1,num_epochs+1,x_step))
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
    ax[1].set_xticks(range(1,num_epochs+1,x_step))
    # Y-axis
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(min_acc,1)
    # Plotting
    ax[1].plot(range(1,num_epochs+1),history.history['accuracy'], label='Training')
    ax[1].plot(range(1,num_epochs+1),history.history['val_accuracy'], label='Validation')
    ax[1].legend()
    
    
    
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
    print("F1-score  : {:.4f}".format(f1))
    print("MCC       : {:.4f}".format(mcc))
    plot_confusion_matrix(labels, predictions, ["beam","reaction"])
    print()
    
    
###########################################################################################################################
###############     ML Algorithms:                                                                          ###############     
###############     Functions used in the first algorithms tried to approach the problem                    ###############
###############     Logistic Regression, RandomForests and Grid Search, KMeans                              ###############
###########################################################################################################################

def best_cl_km(n_cl, clust, x_train, x_val, labels_train):
    
    if not (isinstance(n_cl, int) or n_cl<2):
        sys.exit('Cluster number not fine!')
    

    """This function finds the best mapping form k-means cluster
    to beam or reaction event for a given number of clusters
    
    Arguments:
        n_cl : number of clusters used in k-means
        clust : fitted n-cluster kmeans method
        x_train : training features
        x_val : validation features
        labels_train : training labels
    Returns:
        KM_pred_train : optimal kmeans predictions for training set
        KM_pred_val : optimal kmeans predictions for validation set
        assoc : kmeans cluster to class mapping
    """
    
    n_cmb = 2**n_cl-2 # possible combinations of assignments clusters to labels
    
    KM_pred_train = clust.predict(x_train)
    KM_pred_train_cmb = np.zeros((len(KM_pred_train), n_cmb))
    
    KM_pred_val = clust.predict(x_val)
    KM_pred_val_cmb = np.zeros((len(KM_pred_val), n_cmb))    
    
    #Now we need to find out which cluster is which type of event
    #We select the one that gives best accuracy on training set
    #for this we loop over all combinations without trivial ones (all beam or all reaction)
    #we denote accuracy_train_010 if cluster 0--> labels 0, cl_1--> lb_1 and cl_2 --> lb_0

    cl_ass = np.zeros((n_cmb,n_cl))
    acc_train = np.zeros(n_cmb)
    acc_val = np.zeros(n_cmb)
    # mapping clusters = 0,1,2 in different combination of labels = 0,1
    # there are 6 combination possible listed below
    
    # Calculate possible Cluster to labels assignment 
    for cmb in np.arange(1,n_cmb+1):
        for binary in range(n_cl):
            cl_ass[cmb-1,binary] = int(format(cmb, 'b').zfill(n_cl)[binary])
        
    # Convert cluster in labels, as dictated from cl_ass
    for cmb in range(n_cmb): 
        for i in range(len(KM_pred_train)):
            for cl_value in range(n_cl):
                if KM_pred_train[i]== cl_value:
                    KM_pred_train_cmb[i,cmb]= cl_ass[cmb,cl_value]

    for cmb in range(n_cmb): 
        for i in range(len(KM_pred_val)):
            for cl_value in range(n_cl):
                if KM_pred_val[i]== cl_value:
                    KM_pred_val_cmb[i,cmb]= cl_ass[cmb,cl_value]
                    
        # Calculate accuracy for every cl_ass
    for cmb in range(n_cmb): 
        acc_train[cmb] = accuracy_score(labels_train, KM_pred_train_cmb[:,cmb])
   
    #Finds best accuracy model
    max_accuracy_KM = np.amax(acc_train)
    max_index_KM = np.where(acc_train == np.amax(acc_train))[0]
    
    # Printing results
    print("KMeans with %i clusters performance:"%n_cl)
    print("Max accuracy obtained is {:.4f}".format(max_accuracy_KM), " using the combination number :", max_index_KM[0]+1)
    assoc = list(zip(np.arange(n_cl), cl_ass[max_index_KM[0],:]))
    print('Combination %i has the "Cluster to Lables" association = ' %(max_index_KM[0]+1) + str(assoc) )
    print("----------------------------------------------------------")
    
    assoc = np.array(assoc)
    KM_pred_train = KM_pred_train_cmb[:,max_index_KM]
    KM_pred_val = KM_pred_val_cmb[:,max_index_KM]
    
    return KM_pred_train, KM_pred_val, assoc


##################################################################################################
###############     CNN Section                                                    ###############     
###############     Function used in the section: Convolutional Neural Network     ###############
###############     These function deals meanly with images manipulations          ###############
##################################################################################################

def prepare_data(DataList):
    
    
    x_values_List = []
    y_values_List = []
    
    for i in range(len(DataList)): # loop on event number
        for j in range(len(DataList[i])): # loop on event rows
            x_values_List.append(DataList[i][j][0]) 
            y_values_List.append(DataList[i][j][1])   
            
    xy_values = np.array((x_values_List,y_values_List))
    

    points = []

    # Convert the data in points , which contain only x,y and pixel value.
    for i in range(len(DataList)): # loop on event number
        points.append([])
        for j in range(len(DataList[i])): # loop on event rows
            points[i].append([])
            for ax in range(2):
                points[i][j].append(DataList[i][j][ax]) # x,y values

    return points





def show_grid(points, x_lim, y_lim, x_spc, y_spc, x_shift=0, y_shift=0):
    """ This function gets the grid parameters in input,
    show the grid info, and plot the grid. 
    In this way it is possible to choose whehter keep or modifiyng the grid,
    before generate the image.
    The grid is squared, and has the same dimension all along the axis.
    x_fea, y_fea = points.
    x_lim, y_lim = extreme of the grid
    x_spc, y _spc = grid spacing (width and height of the single grid square)
    x_shift, y _shift = is the grid symmetric respect the origin? if not, insert the shift in respect to the origin
    """
      
    # Calculate number of pixel
    
    # X-direction
    x_pxl = math.ceil((x_lim+abs(x_shift))/x_spc)*2 
    y_pxl = math.ceil((y_lim+abs(y_shift))/y_spc)*2 
    
    print("Grid Parameters:")
    print("----------------------------------")
    
    # X information
    print("Pixeling over x-direction:")
    print("X grid spacing: ", x_spc)
    print("First x cell limits: ", (x_shift, x_spc+x_shift))
    print("Number of pixel on x direction: %i*2 = "%(x_pxl/2), x_pxl)
    
    # Y information
    print("Pixeling over y-direction:")
    print("X grid spacing: ", y_spc)
    print("First y cell limits: ", (y_shift, y_spc+y_shift))
    print("Number of pixel in y direction: %i*2 = "%(y_pxl/2), y_pxl)

    
    # Generate arrays to perform math operations
    x_values_List = []
    y_values_List = []

    
    for i in range(len(points)): # loop on event number
        for j in range(len(points[i])): # loop on event rows
            x_values_List.append(points[i][j][0]) 
            y_values_List.append(points[i][j][1])   

    x_values = np.array(x_values_List)
    y_values = np.array(y_values_List)

    
    print("Showing the Pixel Grid") 
    # Using Pad Grid (Grid used in converting in picture)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x_values,y_values, c = "black", alpha=0.8)
    ax.set_title("Sum of All Events")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    ax.set_xticks(np.arange(+x_shift-x_spc*5, -x_shift+x_spc*5, x_spc))
    ax.set_yticks(np.arange(+y_shift-y_spc*5, -y_shift+y_spc*5, y_spc))
    ax.set_xlim(-10,+10)
    ax.set_ylim(-10,+10) # I want to maintan plot symmetric

    plt.grid(color='blue', linestyle='-', linewidth=1)
    
    fig.tight_layout()
    plt.show()
    
    
    
def generate_images(points, pixel_values, x_lim, y_lim, x_spc, y_spc, x_shift, y_shift):
    
    x_pxl = math.ceil((x_lim+abs(x_shift))/x_spc)*2 
    y_pxl = math.ceil((y_lim+abs(y_shift))/y_spc)*2 
    
    images = np.zeros((len(points),y_pxl,x_pxl))
    
    for i in range(len(points)): # loop on event number
        for j in range(len(points[i])): # loop on event rows
            n_y = math.floor((points[i][j][1]-x_shift)/y_spc) + round(y_pxl/2) # calculate which y image pixel fired
            n_x = math.floor((points[i][j][0]-y_shift)/x_spc) + round(x_pxl/2) # calculate which x image pixel fired
   
            images[i,n_y,n_x] =  images[i,n_y,n_x] + pixel_values[i][j]
        
    return images 



def reduce_images_dim(images, dim):
    
    images_r = np.zeros((images.shape[0],dim,dim))
    
    for i in range(images.shape[0]): 
        for j in range(dim): #y
            for h in range(dim): # x
                y_idx = round(images.shape[1]/2)-round(dim/2) +j
                x_idx = round(images.shape[2]/2)-round(dim/2) +h
                images_r[i,j,h] =  images[i, y_idx, x_idx]  
    return images_r    



def fuse_pixels(images,fuse_x,fuse_y):
    
    y_pxl_new = math.ceil(images.shape[1]/fuse_y)
    x_pxl_new = math.ceil(images.shape[2]/fuse_x)
    
    print(y_pxl_new,x_pxl_new)
    images_f = np.zeros((images.shape[0],y_pxl_new,x_pxl_new))
    print(images_f.shape)
    for i in range(images.shape[0]): 
        for j in range(images.shape[1]): #y
            for h in range(images.shape[2]): # x
                n_y_new = math.floor(j/fuse_y)
                n_x_new = math.floor(h/fuse_x)
                
                images_f[i][n_y_new][n_x_new] = images_f[i][n_y_new][n_x_new] + images[i][j][h]
    return images_f  


    
def normalize_image_data(images):
    """ Takes an imported set of images and normalizes values to between
    0 and 255 using min-max scaling across the whole image set.
    Arguments:
        images : 2d image with amplitude
    Returns:
        images : min-max scaled images
    """
    img_max = np.amax(images)
    img_min = np.amin(images)
    #Debug 
    #print('While executing "normalize_image_data":')
    #print("The max value of images is: ", img_max, " while the minimum is: ", img_min)
    if img_max==0:
        print("Error: File given is made by black images (only zeros)")
    else: 
        if (img_max - img_min) > 0:
            images = np.around( 255 * (images - img_min) / (img_max - img_min))
        else: 
            images = 0
            print("Error: File given is made by same values images, now it has been normalized to 1")
            
    return images



def plot_images(images, labels, plot_row=3, idx=0):

    fig, ax = plt.subplots(plot_row,2,figsize=(18, plot_row*7))
    for i in range(plot_row):
        for j in range(2): 
            my_pic=ax[i][j].imshow(images[idx],vmin=0, vmax=255, cmap='inferno')
            if labels[idx]>0.5:
                ax[i][j].set_title("Image "+ str(idx) + ": Reaction Event")
            else:
                ax[i][j].set_title("Image "+ str(idx) + ": Beam Event")
            
            ax[i][j].set_xlabel("Pixel X")
            ax[i][j].set_ylabel("Pixel Y")
        

            idx = idx +1
            cbar = fig.colorbar(my_pic, ax= ax[i][j], extend='both')
            cbar.minorticks_on()

    fig.tight_layout() # adjust automatically spacing between sublots
    plt.show()
    
    
    
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
    output = layers.Dense(1, activation=tf.nn.sigmoid)(net)

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
   

#########################################################################################################
###############     Dimensionality Reduction Section                                      ###############     
###############     Function used in the section: Dimensionality Reduction Algorithms     ###############
#########################################################################################################


    
def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.5, contour=True):
    """Function taken from: https://github.com/ageron/handson-ml2/blob/master/07_ensemble_learning_and_random_forests.ipynb
       Plots the decision boundary for a classifier on a 2d feature set.
    Arguments:
        clf : fitted classifier to be used (needs a predict method)
        X : 2d feature set
        y : labels corresponding to X
        axes : axes limits for plotting
        alpha : opacity of points
        contour : whether to show decision boundary or not.
    Returns:
        none
    """
    from matplotlib.colors import ListedColormap
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    X_new = np.array(X_new, dtype=np.double)
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=0.8)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

def make_2d_vis(xSimple_train_PCA,xSimple_train_TSNE,Labels_train):
    
    """This function creates the visualizations of different models
    trained on the 2d PCA and TSNE reduced dataset.
    Adapted from: https://github.com/ageron/handson-ml2
    
    Arguments:
        xSimple_train_PCA : PCA features of training set
        xSimple_train_TSNE : t-SNE features of training set
        labels_train : training labels
    Returns:
        none
    """
    import matplotlib.pyplot as plt
    #First we fit the models
    from sklearn.linear_model import LogisticRegression
    
    logreg_PCA = LogisticRegression()
    logreg_PCA.fit(xSimple_train_PCA, Labels_train)
    LR_pred_train_PCA = logreg_PCA.predict(xSimple_train_PCA)
    
    logreg_TSNE = LogisticRegression()
    logreg_TSNE.fit(xSimple_train_TSNE, Labels_train)
    LR_pred_train_TSNE = logreg_TSNE.predict(xSimple_train_TSNE)
    
    from sklearn.ensemble import RandomForestClassifier
    
    RFC_PCA = RandomForestClassifier()
    RFC_PCA.fit(xSimple_train_PCA, Labels_train)
    RFC_pred_train_PCA = RFC_PCA.predict(xSimple_train_PCA)
    
    RFC_TSNE = RandomForestClassifier()
    RFC_TSNE.fit(xSimple_train_TSNE, Labels_train)
    RFC_pred_train_TSNE = RFC_TSNE.predict(xSimple_train_TSNE)
    
    from sklearn.cluster import KMeans
    
    KM2_PCA = KMeans(n_clusters=2)
    KM2_PCA.fit(xSimple_train_PCA)
    KM2_pred_train_PCA = KM2_PCA.predict(xSimple_train_PCA)
    
    KM2_TSNE = KMeans(n_clusters=2)
    KM2_TSNE.fit(xSimple_train_TSNE)
    KM2_pred_train_TSNE = KM2_TSNE.predict(xSimple_train_TSNE)
    
    from sklearn import svm
    
    SVM_PCA = svm.SVC()
    SVM_PCA.fit(xSimple_train_PCA, Labels_train)
    SVM_pred_train_PCA = SVM_PCA.predict(xSimple_train_PCA)
    
    SVM_TSNE = svm.SVC()
    SVM_TSNE.fit(xSimple_train_TSNE, Labels_train)
    SVM_pred_train_TSNE = SVM_TSNE.predict(xSimple_train_TSNE)
    
    
    #Make a figure with subplots
    fig, ax = plt.subplots(3, 2, figsize=(18, 24))
    

    plt.sca(ax[0][0])
    plot_decision_boundary(logreg_PCA, xSimple_train_PCA, Labels_train, axes=[-3.1,5.2,-4,6])
    ax[0][0].text(4, 3, "Reaction Events", fontsize=14, color="b", ha="center")
    ax[0][0].text(-2, 1.8, "Beam Events", fontsize=14, color="orange", ha="center")
    ax[0][0].set_title("Logistic regression after PCA", fontsize=18)

    
   
    plt.sca(ax[0][1])
    plot_decision_boundary(logreg_TSNE, xSimple_train_TSNE, Labels_train, axes=[-60,60,-50,60])
    ax[0][1].text(40, 45, "Reaction Events", fontsize=14, color="b", ha="center")
    ax[0][1].text(-40, 20, "Beam Events", fontsize=14, color="orange", ha="center")
    ax[0][1].set_title("Logistic regression after t-SNE", fontsize=18)

    
    plt.sca(ax[1][0])
    plot_decision_boundary(RFC_PCA, xSimple_train_PCA, Labels_train, axes=[-3.1,5.2,-4,6])
    ax[1][0].text(4, 3, "Reaction Events", fontsize=14, color="b", ha="center")
    ax[1][0].text(-2, 1.8, "Beam Events", fontsize=14, color="orange", ha="center")
    ax[1][0].set_title("Random forest after PCA", fontsize=18)
                           
    plt.sca(ax[1][1])
    plot_decision_boundary(RFC_TSNE, xSimple_train_TSNE, Labels_train, axes=[-60,60,-50,60])
    ax[1][1].text(40, 45, "Reaction Events", fontsize=14, color="b", ha="center")
    ax[1][1].text(-40, 20, "Beam Events", fontsize=14, color="orange", ha="center")
    ax[1][1].set_title("Random forest after t-SNE", fontsize=18)
    
    plt.sca(ax[2][0])
    plot_decision_boundary(SVM_PCA, xSimple_train_PCA, Labels_train, axes=[-3.1,5.2,-4,6])
    ax[2][0].text(4, 3, "Reaction Events", fontsize=14, color="b", ha="center")
    ax[2][0].text(-2, 1.8, "Beam Events", fontsize=14, color="orange", ha="center")
    ax[2][0].set_title("Support vector machine after PCA", fontsize=18)
                           
    plt.sca(ax[2][1])
    plot_decision_boundary(SVM_TSNE, xSimple_train_TSNE, Labels_train, axes=[-60,60,-50,60])
    ax[2][1].text(40, 45, "Reaction Events", fontsize=14, color="b", ha="center")
    ax[2][1].text(-40, 20, "Beam Events", fontsize=14, color="orange", ha="center")
    ax[2][1].set_title("Support vector machine after t-SNE", fontsize=18)
    
    #KMeans does not work properly
    #plt.sca(ax[3][0])
    #plot_decision_boundary(KM2_PCA, xSimple_train_PCA, Labels_train, axes=[-3.1,5.2,-4,6])
    #ax[3][0].text(4, 3, "Reaction Events", fontsize=14, color="b", ha="center")
    #ax[3][0].text(-2, 1.8, "Beam Events", fontsize=14, color="orange", ha="center")
    #ax[3][0].set_title("K-means (2 clusters) after PCA", fontsize=18)
    #ax[3][0].scatter(KM2_PCA.cluster_centers_[:,0], KM2_PCA.cluster_centers_[:,1], marker='x', s=50, linewidths=50,
    #        color='k', zorder=11, alpha=1)
                           
    #plt.sca(ax[3][1])
    #plot_decision_boundary(KM2_TSNE, xSimple_train_TSNE, Labels_train, axes=[-60,60,-50,60])
    #ax[3][1].text(40, 45, "Reaction Events", fontsize=14, color="b", ha="center")
    #ax[3][1].text(-40, 20, "Beam Events", fontsize=14, color="orange", ha="center")
    #ax[3][1].set_title("K-means (2 clusters) after t-SNE", fontsize=18)
    #ax[3][1].scatter(KM2_TSNE.cluster_centers_[:,0], KM2_TSNE.cluster_centers_[:,1], marker='x', s=50, linewidths=50,
     #       color='k', zorder=11, alpha=1)
    

def make_2d_vis_autoencoder(xt,yt,Labels_train):
    
    """This function creates the visualizations of different models
    trained on the 2d PCA and TSNE reduced dataset.
    Adapted from: https://github.com/ageron/handson-ml2
    
    Arguments:
        xSimple_train_PCA : PCA features of training set
        xSimple_train_TSNE : t-SNE features of training set
        labels_train : training labels
    Returns:
        none
    """
    dataset = []

    for x,y in zip(xt,yt):

        dataset.append(np.array([x,y]))

    dataset = np.array(dataset)


    import matplotlib.pyplot as plt
    #First we fit the models
    from sklearn.linear_model import LogisticRegression
   
    logreg_autoencoder = LogisticRegression()
    logreg_autoencoder.fit(dataset, Labels_train)
    LR_pred_train_autoencoder = logreg_autoencoder.predict(dataset)

    from sklearn.ensemble import RandomForestClassifier

    RFC_autoencoder = RandomForestClassifier()
    RFC_autoencoder.fit(dataset, Labels_train)
    RFC_pred_train_autoencoder = RFC_autoencoder.predict(dataset)

    from sklearn.cluster import KMeans

    KM2_autoencoder = KMeans(n_clusters=2)
    KM2_autoencoder.fit(dataset)
    KM2_pred_train_autoencoder = KM2_autoencoder.predict(dataset)

    from sklearn import svm

    SVM_autoencoder = svm.SVC()
    SVM_autoencoder.fit(dataset, Labels_train)
    SVM_pred_train_autoencoder = SVM_autoencoder.predict(dataset)


    #Make a figure with subplots
    fig, ax = plt.subplots(3, figsize=(18, 24))

    plt.sca(ax[0])
    plot_decision_boundary(logreg_autoencoder, dataset, Labels_train)
    ax[0].text(1, -0.5, "Reaction Events", fontsize=14, color="b", ha="center")
    ax[0].text(-1, 1, "Beam Events", fontsize=14, color="orange", ha="center")
    ax[0].set_title("Logistic regression after Autoencoder", fontsize=18)

    plt.sca(ax[1])
    plot_decision_boundary(RFC_autoencoder, dataset, Labels_train)
    ax[1].text(1, -0.5, "Reaction Events", fontsize=14, color="b", ha="center")
    ax[1].text(1, 1, "Beam Events", fontsize=14, color="orange", ha="center")
    ax[1].set_title("Random forest after Autoencoder", fontsize=18)

    plt.sca(ax[2])
    plot_decision_boundary(SVM_autoencoder, dataset, Labels_train)
    ax[2].text(1, -0.5, "Reaction Events", fontsize=14, color="b", ha="center")
    ax[2].text(-0.5, 0.7, "Beam Events", fontsize=14, color="orange", ha="center")
    ax[2].set_title("Support vector machine after Autoencoder", fontsize=18)

        
def plot_encoder_net(x,y):
    
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.title("Training Set Latent representation", fontsize=20)

    plt.scatter(x, y, c = 'black')

    plt.xlabel('X Latent Space', fontsize=18)
    ax.set_xticks(np.arange(-1,1,0.2))
    ax.set_xlim(-1,+1)

    plt.ylabel('Y Latent Space', fontsize=18)
    ax.set_yticks(np.arange(-1,1,0.2))
    ax.set_ylim(-1,+1)

    fig.tight_layout()
    plt.show()
    
def plot_kmeans_clustering(encoder_pred, Labels_train, clusters, assoc, title):
    
    #Assign true label to (x,y) points in latent space
    get_true_reaction = encoder_pred[Labels_train > 0.5]
    get_true_beam = encoder_pred[Labels_train < 0.5]


    # Ask for centroid positions in latent space
    centroids = clusters.cluster_centers_

    # Draw maps of KMeans predictions as Background class regions
    points = np.random.uniform(-1,1,(10,2))

    p_reac_x = []
    p_reac_y = []
    p_beam_x = []
    p_beam_y = []

    for x in np.arange(-1, 1, 0.01): 
        for y in np.arange(-1, 1, 0.01):
            p = np.array([x,y], dtype=np.float32).reshape(1,-1)
            p_cluster = int(clusters.predict(p)) # determine which cluster p belongs to
            p_label = assoc[p_cluster,1] # determine which label p must be assigned to

            if(p_label > 0.5 ):
                p_reac_x.append(p[0][0])
                p_reac_y.append(p[0][1])
            else:            
                p_beam_x.append(p[0][0])
                p_beam_y.append(p[0][1])

    
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.title(title, fontsize=20);

    # Plotting dots : color is true label
    plt.scatter(get_true_reaction[:,0], get_true_reaction[:,1], color='gray', label='Reaction', alpha=0.8)
    plt.scatter(get_true_beam[:,0], get_true_beam[:,1], color='red', label='Beam', alpha=0.5)

    # Plotting prediction area of k means
    plt.scatter(p_reac_x, p_reac_y, color='gray', marker='s', s=5, alpha=0.25)
    plt.scatter(p_beam_x, p_beam_y, color='red', marker='s', s=5, alpha=0.25)

    # Plotting KMeans centroids
    for idx,c in enumerate(centroids):
         plt.scatter([centroids[idx][0]],[centroids[idx][1]], marker='*', s=500, c='black')


    plt.xlabel('X Latent Space', fontsize=18)
    ax.set_xticks(np.arange(-1,1,0.2))
    ax.set_xlim(-1,+1)

    plt.ylabel('Y Latent Space', fontsize=18)
    ax.set_yticks(np.arange(-1,1,0.2))
    ax.set_ylim(-1,+1)

    plt.legend(fontsize=20);
    #plt.xticks(fontsize=15)
    #plt.yticks(fontsize=15)

    fig.tight_layout()
    plt.show()