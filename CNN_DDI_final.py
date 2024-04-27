# from NLPProcess import NLPProcess
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
import csv
import sqlite3
import time
import numpy as np
import pandas as pd
import hashlib
import os, gc
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

import tensorflow as tf

def set_max_gpu_mem(size=10,unit=1024):
  limit = size * unit
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
      try:
          # Restrict TensorFlow to only allocate specified memory on the first GPU
          print(f"Setting GPU memory limit to {size}GB.")
          tf.config.experimental.set_virtual_device_configuration(
              gpus[0],
              [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)])
          logical_gpus = tf.config.experimental.list_logical_devices('GPU')
          print(len(gpus), "Physical GPU,", len(logical_gpus), f"Logical GPU(s) with memory limit:{limit}")
      except RuntimeError as e:
          # Virtual devices must be set before GPUs have been initialized
          print(e)

set_max_gpu_mem()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
 
def print_memory_usage(unit_size=10 ** 6):
    """Prints current memory usage stats.
    See: https://stackoverflow.com/a/15495136

    :return: None
    """ 
    import psutil,os
    PROCESS = psutil.Process(os.getpid())
    total, available, percent, used, free, *_ = psutil.virtual_memory()
    total, available, used, free = total / unit_size, available / unit_size, used / unit_size, free / unit_size
    proc = PROCESS.memory_info()[1] / unit_size
    print('process = %s total = %s available = %s used = %s free = %s percent = %s'
          % (proc, total, available, used, free, percent))
    
output_folder = 'CNN_DDI'
BASE_PATH = f"./{output_folder}/"
# Define data and model path
EVENT_DATA_PATH = BASE_PATH + "event.db"
# DRUG_LIST_PATH =  BASE_PATH + "DrugList.txt"
WEIGHT_PATH =  BASE_PATH + "models_weights"

VECTOR_SIZE = 572                       # num drugs (model input size)
EVENT_NUM = 65                          # num unique event types
DROP_RATE = 0.3                         # Default dropout rate


def DNN(vector_size=VECTOR_SIZE, event_num=EVENT_NUM, drop_rate=DROP_RATE):
    """
    A deep neural network (DNN) model for predicting drug-drug interactions.

    original source code from:
    https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L35

    Args:
        vector_size (int): The size of the input feature vector.
        event_num (int): The number of unique interaction events (classes) to predict.
        drop_rate (float): The dropout rate for regularization.

    Returns:
        model: A compiled Keras model ready for training.
    """
    # Define the input layer
    train_input = Input(shape=(vector_size * 2,), name='Inputlayer')
     # First dense layer with 512 units and ReLU activation
    train_in = Dense(512, activation='relu')(train_input)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(drop_rate)(train_in)
    # Second dense layer with 256 units and ReLU activation
    train_in = Dense(256, activation='relu')(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(drop_rate)(train_in)
    # Output dense layer with 'event_num' units for classification
    train_in = Dense(event_num)(train_in)
    # Softmax activation to convert logits to probabilities for multi-class classification
    out = Activation('softmax')(train_in)
    # Create the model
    model = Model(inputs=train_input, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# from keras.models import Model
# from keras.layers import Input, Dense, Conv1D, BatchNormalization, Activation, Flatten, Dropout, Add
from tensorflow.keras.layers import Conv1D, Flatten, Add
from tensorflow.keras.layers import LeakyReLU
def CNN_DDI(vector_size=VECTOR_SIZE, event_num=EVENT_NUM, loss_fn='categorical_crossentropy'):
    """
    Convolutional Neural Network (CNN) model for predicting drug-drug interactions (DDIs).

    Implementation based on "CNN-DDI: a learning-based method for predicting drug–drug interactions using convolution neural networks."
    https://doi.org/10.1186/s12859-022-04612-2

    Args:
        vector_size (int): The size of the input feature vector for each drug.
        event_num (int): The number of unique DDI event types to predict.

    Returns:
        model: A compiled Keras model ready for training.
    """
    # Define the input layer
    inputs = Input(shape=(vector_size, 2), name='InputLayer')

    # Convolutional layers as specified in the paper
    conv1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.2)(conv1)

    conv2 = Conv1D(filters=128, kernel_size=3, strides=1, padding='same')(conv1)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    # Residual block starts
    conv3_1 = Conv1D(filters=128, kernel_size=3, strides=1, padding='same')(conv2)
    conv3_1 = LeakyReLU(alpha=0.2)(conv3_1)

    conv3_2 = Conv1D(filters=128, kernel_size=3, strides=1, padding='same')(conv3_1)
    conv3_2 = LeakyReLU(alpha=0.2)(conv3_2)

    # Add the input of the residual block (conv2) to its output (conv3_2)
    res_out = Add()([conv2, conv3_2])
    # Residual block ends

    conv4 = Conv1D(filters=256, kernel_size=3, strides=1, padding='same')(res_out)
    conv4 = LeakyReLU(alpha=0.2)(conv4)

    # Flatten the output of the last convolutional layer
    flatten = Flatten()(conv4)

    # Fully connected layers
    fc1 = Dense(267, activation='relu')(flatten)

    fc2 = Dense(event_num)(fc1)  # Assuming 'num_classes' is the number of DDI event types
    out = Activation('softmax')(fc2)

    # Create the model
    model = Model(inputs=inputs, outputs=out)

    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    return model

# Define the Jaccard Similarity function
def Jaccard(matrix):
    """
    Calculate the Jaccard similarity between rows of a given matrix.

    original source code from:
    https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L89

    Args:
        matrix (array-like): A 2D array (or matrix) where each row represents a set in binary form (1s and 0s),
                             with 1 indicating the presence of an element in the set, and 0 indicating absence.

    Returns:
        numpy.matrix: A matrix of Jaccard similarity scores between each pair of rows in the input matrix.
    """
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return numerator / denominator

def Gaussian(matrix, gamma=None):
    """
    Calculate the Gaussian similarity between rows of a given matrix.

    Args:
        matrix (array-like): A 2D array (or matrix) where each row represents a feature vector.
        gamma (float, optional): The hyperparameter for the Gaussian kernel. If None, it is computed as
                                 1 / (average of the absolute values of the elements across all feature vectors).

    Returns:
        numpy.ndarray: A matrix of Gaussian similarity scores between each pair of rows in the input matrix.
    """
    if gamma is None:
      gamma = 1.0 / np.mean(np.abs(matrix))

    # Compute the squared Euclidean distance between each pair of rows
    sq_dists = np.sum((matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]) ** 2, axis=2)

    # Compute the Gaussian similarity
    return np.exp(-gamma * sq_dists)

# Define the Cosine Similarity function
def Cosine(matrix):
    """
    Calculate the Cosine similarity between rows of a given matrix.

    Args:
        matrix (array-like): A 2D array (or matrix) where each row represents a feature vector.

    Returns:
        numpy.ndarray: A matrix of Cosine similarity scores between each pair of rows in the input matrix.
    """
    normalized_matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.dot(normalized_matrix, normalized_matrix.T)

def Kulczynski(matrix):
    """
    Calculate the Kulczynski similarity between rows of a given matrix.

    Args:
        feature_matrix (np.array): A 2D binary array where each row represents a feature vector.

    Returns:
        np.array: A matrix of Kulczynski similarity scores between each pair of rows in the input matrix.
    """
    # Calculate the intersection for each pair of feature vectors
    intersection = np.dot(matrix, matrix.T)

    # Calculate the sum of features for each vector
    sum_features = matrix.sum(axis=1)

    # Avoid division by zero in case of empty feature vectors
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate the Kulczynski similarity
        similarity = 0.5 * ((intersection / sum_features[:, None]) + (intersection / sum_features[None, :]))
        # Replace NaNs with 0 for cases where division by zero occurs
        similarity = np.nan_to_num(similarity)

    return similarity

similarity_fn_map = {'Jaccard': Jaccard,
                     'Gaussian': Gaussian,
                     'Cosine': Cosine,
                     'Kulczynski': Kulczynski}


def feature_vector(feature_name, df, vector_size, similarity_measure='Jaccard'):
    """
    Generates a feature vector for each drug based on the specified feature using Jaccard Similarity and PCA reduction.

    This function first constructs a feature matrix for drugs based on the presence or absence of specific features
    (e.g., targets, enzymes). It then computes the Jaccard Similarity matrix for these drugs and finally reduces the
    dimensionality of this matrix to the specified vector size using PCA.

    original source code from:
    https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L86

    Args:
        feature_name (str): The name of the feature column in the DataFrame `df` to be used for generating feature vectors.
        df (DataFrame): A pandas DataFrame containing drug data. Each row corresponds to a drug, and the specified
                        feature column contains feature identifiers separated by '|'.
        vector_size (int): The target number of dimensions for the feature vectors after PCA reduction.

    Returns:
        numpy.ndarray: A 2D array where each row represents the reduced-dimensionality feature vector for a drug.
    """
    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # Extract unique features from the feature column for all drugs
    for features in drug_list:
        for each_feature in features.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)

    # Initialize a feature matrix with zeros
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    # Construct a DataFrame for easier manipulation
    df_feature = DataFrame(feature_matrix, columns=all_feature)

    # Populate the feature matrix with 1s where a drug has a particular feature
    for i, features in enumerate(drug_list):
        for each_feature in features.split('|'):
            df_feature.at[i, each_feature] = 1

    # Compute the Similarity matrix
    Similarity = similarity_fn_map[similarity_measure]
    sim_matrix = Similarity(np.array(df_feature))
    sim_matrix = np.asarray(sim_matrix)

    # Apply PCA to reduce the dimensionality of the similarity matrix
    pca = PCA(n_components=vector_size)
    pca.fit(sim_matrix)
    reduced_sim_matrix = pca.transform(sim_matrix)

    return reduced_sim_matrix

def prepare(df_drug, feature_list, vector_size, mechanism, action, drugA, drugB, similarity_measure='Jaccard'):
    """
    Prepares feature vectors and labels for drug interaction events.

    This function processes a list of drug interaction features to generate corresponding
    feature vectors and labels. It assigns a unique numerical label to each unique
    mechanism-action pair and constructs feature vectors for each drug based on the provided
    features.

    source code adapted from:
    https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L50

    Args:
        df_drug (DataFrame): DataFrame containing drug data, including names.
        feature_list (list): List of features to be included in the feature vector.
        vector_size (int): The size of the feature vector for each feature.
        mechanism (Series): Series of mechanisms involved in drug interactions.
        action (Series): Series of actions resulting from drug interactions.
        drugA (Series): Series of primary drugs involved in interactions.
        drugB (Series): Series of secondary drugs involved in interactions.

    Returns:
        tuple: A tuple containing:
            - new_feature (numpy.ndarray): Array of feature vectors for drug interactions.
            - new_label (numpy.ndarray): Array of labels for each drug interaction event.
            - event_num (int): The total number of unique interaction events.
    """
    d_label = {}
    d_feature = {}
    d_event = []

    # Concatenate mechanism and action to form unique interaction events
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])

    # Count occurrences of each event and assign a unique label
    count = {}
    for event in d_event:
        count[event] = count.get(event, 0) + 1
    sorted_events = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i, (event, _) in enumerate(sorted_events):
        d_label[event] = i

    # Initialize a zero vector for feature aggregation
    vector = np.zeros((len(df_drug['name']), 0), dtype=float)

    # Aggregate feature vectors for each feature in the list
    for feature in feature_list:
        vector = np.hstack((vector, feature_vector(feature, df_drug, vector_size,similarity_measure)))

    # Map drug names to their feature vectors
    for i, name in enumerate(df_drug['name']):
        d_feature[name] = vector[i]

    # Construct feature vectors and labels for each interaction event
    new_feature = []
    new_label = []
    for i in range(len(d_event)):
        combined_feature = np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))
        new_feature.append(combined_feature)
        new_label.append(d_label[d_event[i]])

    new_feature = np.array(new_feature)
    new_label = np.array(new_label)
    event_num = len(sorted_events)

    return (new_feature, new_label, event_num)

def logistic_regression_pred(X_train, Y_train, X_test):
    #Logistic Regression model
    # original source code from: https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L182
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    pred = model.predict_proba(X_test)
    return pred

def random_forest_pred(X_train, Y_train, X_test):
    #Random Forest Classifier with 100 trees
    # original source code from: https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L172
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, Y_train)
    pred = model.predict_proba(X_test)
    return pred

def gbdt_pred(X_train, Y_train, X_test):
    #Gradient Boosting Decision Tree (GBDT) model
    # original source code from: https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L174
    model = GradientBoostingClassifier()
    model.fit(X_train, Y_train)
    pred = model.predict_proba(X_test)
    return pred

def svm_pred(X_train, Y_train, X_test):
    #Support Vector Machine (SVM) model with probability estimates
    # original source code from: https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L176
    model = SVC(probability=True)
    model.fit(X_train, Y_train)
    pred = model.predict_proba(X_test)
    return pred

def knn_pred(X_train, Y_train, X_test):
    #K-Nearest Neighbors (KNN) classifier with 4 neighbors
    # original source code from: https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L180
    model = KNeighborsClassifier(n_neighbors=4)
    model.fit(X_train, Y_train)
    pred = model.predict_proba(X_test)
    return pred

def get_index(label_matrix, event_num, seed, CV):
    """
    Generate indices for K-fold cross-validation for each class in the label matrix.

    original source code from:
    https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L117

    Args:
        label_matrix (array-like): A 1D array containing the class labels for each sample.
        event_num (int): The number of unique events or classes.
        seed (int): Random seed for reproducibility of the shuffle in KFold.
        CV (int): The number of folds for the K-fold cross-validation.

    Returns:
        numpy.ndarray: An array of indices indicating the fold number for each sample.
    """
    # Initialize an array to store the fold indices for all samples
    index_all_class = np.zeros(len(label_matrix))
    # generate fold indices for each class
    for j in range(event_num):
        # Find the indices of samples belonging to the current class
        index = np.where(label_matrix == j)
        # Initialize KFold with the specified number of splits, shuffling, and random seed
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
        # Initialize a counter for the fold number
        k_num = 0
        # Get train and test indices for each fold
        for train_index, test_index in kf.split(range(len(index[0]))):
            # Assign the fold number to the corresponding samples in the overall index array
            index_all_class[index[0][test_index]] = k_num
            # Increment the fold number
            k_num += 1
    # Return the array of fold indices
    return index_all_class


def cross_validation(feature_matrix, label_matrix, clf_type, event_num, seed, CV, num_epochs, batch_size, patience=10,
                     evalute_only=False, save_weights=True, loss_fn='categorical_crossentropy',weight_path=''):
    """
    Perform K-fold cross-validation to evaluate the performance of specified classifiers on a DDI prediction task.

    original source code from:
    https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L130

    Args:
        feature_matrix (array-like or list of array-like): Feature matrix or list of feature matrices for training and testing.
        label_matrix (array-like): Label matrix corresponding to the true class labels.
        clf_type (str): Type of classifier to be evaluated. Supported types include 'DDIMDL', 'CNN_DDI', 'RF', 'GBDT', 'SVM', 'FM', 'KNN', and logistic regression.
        event_num (int): Number of unique events or classes.
        seed (int): Random seed for reproducibility of the shuffle in KFold.
        CV (int): Number of folds for the K-fold cross-validation.
        num_epochs (int): Number of training epochs for neural network models (DDIMDL and CNN_DDI).
        batch_size (int): Batch size used during training of neural network models (DDIMDL and CNN_DDI).
        patience (int, optional): Number of epochs with no improvement after which training will be stopped for early stopping. Defaults to 10.
        evalute_only (bool): flag to skip CNN training if weights available and run only evaluation
        save_weights (bool): flag to save CNN weights after training  
        loss_fn(str): string to indidcate loss function to use during CNN training. Default is categorical_crossentropy.
        weight_path(str): folder to save model weights.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array contains overall evaluation metrics for the model,
               and the second array contains evaluation metrics for each class.
    """
    # Ensure the directory for saving model weights exist
    if not save_weights:
      if not weight_path: 
        weight_path = generate_weight_path(clf_type=clf_type,CV_seed=seed,num_folds=CV,
                     num_epochs=num_epochs, batch_size=batch_size,loss_fn=loss_fn)
      if not os.path.exists(weight_path):
        print(f'Creating folder: {weight_path}')
        os.makedirs(weight_path)

    # Initialize arrays to store evaluation results
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    # Generate indices for K-fold cross-validation
    index_all_class = get_index(label_matrix, event_num, seed, CV)
    matrix = []
    if type(feature_matrix) != list:
        matrix.append(feature_matrix)
        feature_matrix = matrix
    for k in range(CV):
        # Split data into training and testing sets based on fold index
        train_index = np.where(index_all_class != k)
        test_index = np.where(index_all_class == k)
        pred = np.zeros((len(test_index[0]), event_num), dtype=float)
        # Train and predict with each feature matrix (in case of multiple feature matrices)
        for i in range(len(feature_matrix)):
            x_train = feature_matrix[i][train_index]
            x_test = feature_matrix[i][test_index]
            y_train = label_matrix[train_index]
            y_test = label_matrix[test_index]
            # one-hot encoding training labels
            y_train_one_hot = np.array(y_train)
            y_train_one_hot = (np.arange(event_num) == y_train[:, None]).astype(dtype='float32')
            # one-hot encoding of testing labels
            y_test_one_hot = np.array(y_test)
            y_test_one_hot = (np.arange(event_num) == y_test[:, None]).astype(dtype='float32')
            if clf_type == 'DDIMDL':
                dnn = DNN()
                print_memory_usage()
                early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto')
                dnn.fit(x_train, y_train_one_hot, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test_one_hot),
                        callbacks=[early_stopping])
                pred += dnn.predict(x_test)
            elif clf_type == 'CNN_DDI':
                x_train_reshaped = x_train.reshape(-1, VECTOR_SIZE, 2)
                x_test_reshaped = x_test.reshape(-1, VECTOR_SIZE, 2)
                cnn_ddi = CNN_DDI(loss_fn=loss_fn)
                print_memory_usage()
                weights_file = os.path.join(weight_path, f'{clf_type.lower()}_fold_{k+1}_feature_{i+1}.h5')
                if evalute_only and os.path.exists(weights_file):
                  cnn_ddi.load_weights(weights_file)
                  print(f"Loaded weights from {weights_file}. Skipping training.")
                else:
                  print(f"No weights found or loading not requested for fold {k+1}, feature {i+1}. Starting training.")
                  early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto')
                  cnn_ddi.fit(x_train_reshaped, y_train_one_hot, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test_reshaped, y_test_one_hot),
                              callbacks=[early_stopping])
                  if save_weights:
                    cnn_ddi.save_weights(weights_file)
                    print(f"Saved weights to {weights_file}.")
                pred += cnn_ddi.predict(x_test_reshaped)
                del cnn_ddi
            elif clf_type == 'RF':
                pred += random_forest_pred(x_train, y_train, x_test)
            elif clf_type == 'GBDT':
                pred += gbdt_pred(x_train, y_train, x_test)
            elif clf_type == 'SVM':
                pred += svm_pred(x_train, y_train, x_test)
            elif clf_type == 'FM':
                pred += gbdt_pred(x_train, y_train, x_test)
            elif clf_type == 'KNN':
                pred += knn_pred(x_train, y_train, x_test)
            elif clf_type == 'LR':
                pred += logistic_regression_pred(x_train, y_train, x_test)
            else:
                raise ValueError(f'{clf_type} is not valid')
        # Aggregate predictions from all feature matrices and determine predicted class
        pred_score = pred / len(feature_matrix)
        pred_type = np.argmax(pred_score, axis=1)
        # Accumulate true labels, predicted labels, and predicted scores
        y_true = np.hstack((y_true, y_test))
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))
        del x_train, x_test, y_train, y_test, y_train_one_hot, y_test_one_hot
        tf.keras.backend.clear_session()
        gc.collect()
    # Evaluate the performance of the classifier
    result_all, result_eve = evaluate(y_pred, y_score, y_true, event_num)
    return result_all, result_eve


def evaluate(pred_type, pred_score, y_test, event_num):
    """
    Evaluate the performance of predictions for multi-class classification.

    original source code from:
    https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L203

    Args:
        pred_type (array-like): Predicted labels for each sample.
        pred_score (array-like): Predicted scores or probabilities for each class for each sample.
        y_test (array-like): True labels for each sample.
        event_num (int): Number of distinct events or classes.

    Returns:
        list: A list containing two numpy arrays. The first array contains overall evaluation metrics for the model,
              and the second array contains evaluation metrics for each class.
    """
    # Define the number of evaluation metrics for overall performance
    all_eval_type = 11
    # Initialize an array to store overall evaluation metrics
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    # Define the number of evaluation metrics for each class
    each_eval_type = 6
    # Initialize an array to store evaluation metrics for each class
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    # Convert true labels to one-hot encoding
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    # Convert predicted labels to one-hot encoding
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))

    # Calculate precision and recall for multi-class classification
    precision, recall, th = multiclass_precision_recall_curve(y_one_hot, pred_score)

    # Calculate overall evaluation metrics
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')

    # Calculate evaluation metrics for each event type
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                         average=None)
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    # Return the overall and per-class evaluation metrics

    return [result_all, result_eve]



def multiclass_precision_recall_curve(y_true, y_score):
    """
    Calculate the precision-recall curve for the first class in a multiclass classification problem.

    This function reshapes the true labels and predicted scores if necessary, and then computes
    the precision-recall curve for the first class. It is designed to work with one-vs-rest
    multiclass classification models where each class is treated independently.

    original source code from:
    https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L265

    Args:
      y_true: array-like of shape (n_samples,) or (n_samples, n_classes)
              True binary labels or binary label indicators for each class.
      y_score: array-like of shape (n_samples,) or (n_samples, n_classes)
               Target scores, can either be probability estimates of the positive class,
               confidence values, or non-thresholded measure of decisions.

    Returns:
      precision: array of shape (n_thresholds + 1,)
                 Precision values such that element i is the precision of predictions with
                 score >= thresholds[i] and the last element is 1.
      recall: array of shape (n_thresholds + 1,)
              Recall values such that element i is the recall of predictions with
              score >= thresholds[i] and the last element is 0.
      pr_thresholds: array of shape (n_thresholds,)
                     Decreasing thresholds on the decision function used to compute
                     precision and recall.
    """
    # Ensure the true labels and scores are 1D arrays, reshaping if necessary
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    # Reshape y_true and y_score to 2D arrays if they are 1D
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))
    # Extract the true labels and scores for the first class
    y_true_c = y_true.take([0], axis=1).ravel()
    y_score_c = y_score.take([0], axis=1).ravel()
    # Compute precision, recall, and thresholds for the first class
    precision, recall, pr_thresholds = precision_recall_curve(y_true_c, y_score_c)
    return (precision, recall, pr_thresholds)

def roc_aupr_score(y_true, y_score, average="macro"):
    """
    Calculate the Area Under the Precision-Recall Curve (AUPR) for binary or multiclass classification.

    original source code from:
    https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L278

    Args:
      y_true: array-like of shape (n_samples,) or (n_samples, n_classes)
              True binary labels or binary label indicators for multiclass classification.
      y_score: array-like of shape (n_samples,) or (n_samples, n_classes)
               Target scores, can either be probability estimates of the positive class,
               confidence values, or non-thresholded measure of decisions.
      average: string, ['micro', 'macro', 'binary'] (default='macro')
               If 'binary', calculate AUPR for binary classification problems.
               If 'micro', calculate metrics globally by considering each element of the label
               indicator matrix as a label.
               If 'macro', calculate metrics for each label, and find their unweighted mean.

    Returns:
      AUPR score: float
                  Area Under the Precision-Recall Curve (AUPR) score.
    """
    # Function to calculate AUPR for binary classification
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    # Function to handle averaging of AUPR scores for multiclass classification
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        # Handle micro averaging
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        # Ensure y_true and y_score are 2D arrays
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        # Calculate AUPR for each class and average
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def convert_overall_results_to_df(result_all):
    """
    Convert the results_all array into a DataFrame with appropriate metric names as column headers.

    Args:
        results_all (numpy array): An array containing overall evaluation metrics for the model.

    Returns:
        DataFrame: A DataFrame containing overall evaluation metrics for the model with named columns.
    """
    metric_names = [
        "Accuracy",
        "AUPR (micro-averaged)",
        "AUPR (macro-averaged)",
        "AUC (micro-averaged)",
        "AUC (macro-averaged)",
        "F1 Score (micro-averaged)",
        "F1 Score (macro-averaged)",
        "Precision (micro-averaged)",
        "Precision (macro-averaged)",
        "Recall (micro-averaged)",
        "Recall (macro-averaged)"
    ]
    result_df = pd.DataFrame(result_all.transpose(), columns=metric_names)
    return result_df

def convert_event_result_to_df(result_eve):
    """
    Convert the result_eve array into a DataFrame with appropriate metric names as column headers.

    Args:
        result_eve (numpy array): An array containing evaluation metrics for each class.

    Returns:
        DataFrame: A DataFrame containing evaluation metrics for each class with named columns.
    """
    metric_names = [
        "Accuracy",
        "AUPR",
        "AUC",
        "F1 Score",
        "Precision",
        "Recall"
    ]
    result_df = pd.DataFrame(result_eve, columns=metric_names)
    return result_df

def save_result(feature_name, result_type, clf_type, result, base_path=BASE_PATH):
    """
    Save the evaluation results of a classifier into a CSV file.

    original source code from:
    https://github.com/YifanDengWHU/DDIMDL/blob/master/DDIMDL.py#L321

    Args:
        feature_name (str): Name of the feature set used for the classifier.
        result_type (str): Type of result being saved (e.g., 'accuracy', 'precision').
        clf_type (str): Type of classifier (e.g., 'CNN-DDI', 'RF').
        result (list): A list of evaluation results to be saved.
        base_path (str, optional): Base path for saving the result file. Defaults to BASE_PATH.

    Returns:
        int: 0 on successful execution.
    """
    # Construct the file path by combining base path, feature name, result type, and classifier type
    file_path = base_path + feature_name + '_' + result_type + '_' + clf_type + '.csv'
    if isinstance(result, pd.DataFrame):
        result.to_csv(file_path, index=False)
    else:
      with open(file_path, "w", newline='') as csvfile:
          writer = csv.writer(csvfile)
          for i in result:
              writer.writerow(i)
    return 0


def generate_weight_path(clf_type=None, similarity_measure=None, feature_name_set=None, 
                         CV_seed=None, num_folds=None, num_epochs=None, batch_size=None, loss_fn=None):
    """
    Generate a path for saving TensorFlow model weights based on a hash of the training arguments.
    """
    unique_identifier='_'.join(list(map(str,filter(None,[feature_name_set, feature_name_set, num_folds, num_epochs, batch_size]))))
    # unique_identifier = f"{feature_name_set}_{CV_seed}_{num_folds}_{num_epochs}_{batch_size}"
    # Generate a hash of the unique identifier
    hash_object = hashlib.sha256(unique_identifier.encode())
    hash_digest = hash_object.hexdigest()[:10] 

    # cnn_ddi_jaccard_categorial_crossentropy_abcde12345
    # folder_name = f"{clf_type}_{similarity_measure}_{loss_fn}_{hash_digest}".lower()
    folder_name='_'.join(list(map(str,filter(None,[clf_type, similarity_measure, loss_fn, hash_digest])))).lower()

    weight_path = os.path.join(WEIGHT_PATH, folder_name)

    os.makedirs(weight_path, exist_ok=True)

    return weight_path

# Main function adjusted for Jupyter Notebook
def main(args):
    seed = 0
    CV = 5
    interaction_num = 10
    # Ensure you have the 'event.db' file accessible in your Google Colab environment.
    # You might need to upload it or access it from Google Drive.
    conn = sqlite3.connect("event.db")
    df_drug = pd.read_sql('select * from drug;', conn)
    df_event = pd.read_sql('select * from event_number;', conn)
    df_interaction = pd.read_sql('select * from event;', conn)

    feature_list = args['featureList']
    featureName = "+".join(feature_list)
    clf_list = args['classifier']
    for feature in feature_list:
        set_name = feature + '+'
    set_name = set_name[:-1]


    drugList = []
    for line in open("DrugList.txt", 'r'):
        drugList.append(line.split()[0])
    if args['NLPProcess'] == "read":
        extraction = pd.read_sql('select * from extraction;', conn)
        mechanism = extraction['mechanism']
        action = extraction['action']
        drugA = extraction['drugA']
        drugB = extraction['drugB']
    else:
        pass
        # mechanism, action, drugA, drugB = NLPProcess(drugList, df_interaction)
    all_matrix = []
    for feature in feature_list:
        print(feature)
        new_feature, new_label, event_num = prepare(df_drug, [feature], VECTOR_SIZE, mechanism, action, drugA, drugB)
        all_matrix.append(new_feature)
    print(len(all_matrix))

    similarity_measure = args['similarity_measure']
    num_folds = args['num_folds']
    num_epochs = args['num_epochs']
    batch_size = args['batch_size']
    evalute_only = args['evaluate_only']
    save_weights =  args['save_weights']
    loss_fn = args['loss_fn']

    result_all = {}
    result_eve = {}
    start = time.perf_counter()
    for clf in clf_list:
      # weight_path=f'{WEIGHT_PATH}/{clf}_{similarity_measure}_{loss_fn}'
      weight_path=generate_weight_path(clf_type=clf, similarity_measure=similarity_measure, 
                                       feature_name_set=featureName, CV_seed=seed, 
                                       num_folds=num_folds, num_epochs=num_epochs,
                                       batch_size=batch_size, loss_fn=loss_fn)

      clf_start = time.perf_counter()
      print(f"running cross validation for {clf}")
      # Perform cross-validation using the specified classifier
      all_result, each_result = cross_validation(all_matrix, new_label, clf, event_num, seed, num_folds, num_epochs, batch_size,
                                                 evalute_only=evalute_only, save_weights=save_weights, loss_fn=loss_fn, weight_path=weight_path)
      clf_end = time.perf_counter()
      timeTaken = clf_end - clf_start
      all_result  = convert_overall_results_to_df(all_result)
      each_result = convert_event_result_to_df(each_result)
      all_result['Time (s)'] = timeTaken
      # Save the cross-validation results to CSV files
      save_result(featureName, 'all', clf, all_result)
      save_result(featureName, 'each', clf, each_result)
      result_all[clf] = all_result
      result_eve[clf] = each_result
      print(all_result)
      print(f"time used for {clf}:", timeTaken )
    print("Total time used:", time.perf_counter() - start)



    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f","--featureList",default=["pathway","target","enzyme", "category"],help="features to use",nargs="+")
    parser.add_argument("-c","--classifier",choices=["CNN_DDI","DDIMDL","RF","KNN","LR"],default=["CNN_DDI"],help="classifiers to use",nargs="+")
    parser.add_argument("-p","--NLPProcess",choices=["read","process"],default="read",help="Read the NLP extraction result directly or process the events again")

    parser.add_argument("-s", "--similarity_measure", choices=["Jaccard", "Cosine", "Gaussian", "Kulczynski"], default="Jaccard", help="similarity measure to use")
    parser.add_argument("-nf", "--num_folds", type=int, default=5, help="number of folds for K-fold cross-validation")
    parser.add_argument("-ne", "--num_epochs", type=int, default=100, help="number of epochs for training")
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("-eo", "--evaluate_only", action='store_true', help="evaluate the model without training")
    parser.add_argument("-sw", "--save_weights", action='store_true', help="save the model weights after training")
    parser.add_argument("-lf", "--loss_fn", choices=["categorical_crossentropy", "kl_divergence", "cosine_similarity","categorical_hinge"], default="categorical_crossentropy", help="loss function to use during training")

    args=vars(parser.parse_args())
    print(args)
    main(args)
    print('done')

