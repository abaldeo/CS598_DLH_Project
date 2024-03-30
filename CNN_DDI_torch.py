# from NLPProcess import NLPProcess
import csv
import sqlite3
import time
import copy
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from numpy.random import seed
from tqdm import tqdm

random_seed=1
seed(random_seed)
torch.manual_seed(random_seed)

event_num = 65
droprate = 0.3
vector_size = 572

NUM_EPOCHS = 100
BATCH_SIZE = 128
CV = 5

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(vector_size * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(droprate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(droprate),
            nn.Linear(256, event_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)



class CNN_DDI(nn.Module):
    def __init__(self):
        super(CNN_DDI, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding='same')
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding='same')
        self.conv3_1 = nn.Conv1d(128, 128, kernel_size=3, padding='same')
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size=3, padding='same')
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding='same')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * vector_size, 267)
        self.fc2 = nn.Linear(267, event_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1, 2) 
        x = self.lrelu(self.conv1(x))
        conv2_out = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3_1(conv2_out))
        x = self.lrelu(self.conv3_2(x))
        x += conv2_out  # Adding the residual connection
        x = self.lrelu(self.conv4(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # def forward(self, x):
    #     x = x.transpose(1, 2)        
    #     x = self.lrelu(self.conv1(x))
    #     x = self.lrelu(self.conv2(x))
    #     residual = x
    #     x = self.lrelu(self.conv3_1(x))
    #     x = self.lrelu(self.conv3_2(x))
    #     x += residual
    #     x = self.lrelu(self.conv4(x))
    #     x = self.flatten(x)
    #     x = torch.relu(self.fc1(x))
    #     x = self.softmax(self.fc2(x))
    #     return x


def prepare(df_drug, feature_list, vector_size,mechanism,action,drugA,drugB):
    d_label = {}
    d_feature = {}
    # Transfrom the interaction event to number
    # Splice the features
    d_event=[]
    for i in range(len(mechanism)):
        d_event.append(mechanism[i]+" "+action[i])
    label_value = 0
    count={}
    for i in d_event:
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    list1 = sorted(count.items(), key=lambda x: x[1],reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]]=i
    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)
    for i in feature_list:
        vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))
    # Transfrom the drug ID to feature vector
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]
    # Use the dictionary to obtain feature vector and label
    new_feature = []
    new_label = []
    name_to_id = {}
    for i in range(len(d_event)):
        new_feature.append(np.hstack((d_feature[drugA[i]], d_feature[drugB[i]])))
        new_label.append(d_label[d_event[i]])
    new_feature = np.array(new_feature)
    new_label = np.array(new_label)
    return (new_feature, new_label, event_num)


def feature_vector(feature_name, df, vector_size):
    # df are the 572 kinds of drugs
    # Jaccard Similarity
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1
    sim_matrix = Jaccard(np.array(df_feature))

    sim_matrix1 = np.array(sim_matrix)
    count = 0
    pca = PCA(n_components=vector_size)  # PCA dimension
    sim_matrix = np.asarray(sim_matrix)
    pca.fit(sim_matrix)
    sim_matrix = pca.transform(sim_matrix)
    return sim_matrix


def get_index(label_matrix, event_num, seed, CV):
    index_all_class = np.zeros(len(label_matrix))
    for j in range(event_num):
        index = np.where(label_matrix == j)
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
        k_num = 0
        for train_index, test_index in kf.split(range(len(index[0]))):
            index_all_class[index[0][test_index]] = k_num
            k_num += 1

    return index_all_class

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device, accumulation_steps=1, patience=10):
    model.to(device)
    best_val_loss = float('inf')
    # best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels) / accumulation_steps  # Normalize loss to account for accumulation
            loss.backward()  # Accumulate gradients

            # optimizer.step()
            running_loss += loss.item() * accumulation_steps  # Correct loss scaling after normalization
            # del inputs, labels, outputs, loss  # Free up memory
            # torch.cuda.empty_cache()  # Clear memory cache
          
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()  # Perform optimization step after accumulating gradients
                optimizer.zero_grad()  # Clear gradients after optimization step

        # Perform optimization step if there are any unflushed gradients (for cases where dataset size is not divisible by accumulation_steps)
        if len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader)}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        pred_list = []
 
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                pred_list.append(outputs.cpu().numpy())

                # Apply torch.max() along dimension 1 to get predicted class indices
                _, predicted = torch.max(outputs, 1)
                
                # Since labels are one-hot encoded, convert them to class indices as well
                _, labels_indices = torch.max(labels, 1)

                total += labels.size(0)
                # print(predicted.shape)
                # print(labels.shape)
                # print(total)
                correct += (predicted == labels_indices).sum().item()
                
            #     del inputs, labels, outputs  # Free up memory
            # torch.cuda.empty_cache()  # Clear memory cache    
            val_loss /= len(val_loader)
            print(f'Validation loss: {val_loss}, accuracy: {100 * correct / total}%')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                early_stop = True
                break    

    # Load best model weights
    # model.load_state_dict(best_model_wts)  

    return np.concatenate(pred_list, axis=0)

def cross_validation(feature_matrix, label_matrix, clf_type, event_num, seed, CV, set_name):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    index_all_class = get_index(label_matrix, event_num, seed, CV)
    matrix = []
    if type(feature_matrix) != list:
        matrix.append(feature_matrix)
        feature_matrix = matrix
    for k in range(CV):
        train_index = np.where(index_all_class != k)
        test_index = np.where(index_all_class == k)
        pred = np.zeros((len(test_index[0]), event_num), dtype=float)
        print(len(feature_matrix))
        for i in range(len(feature_matrix)):
            x_train = feature_matrix[i][train_index]
            x_test = feature_matrix[i][test_index]
            y_train = label_matrix[train_index]
            y_test = label_matrix[test_index]
            y_train_one_hot = torch.nn.functional.one_hot(torch.tensor(y_train), num_classes=event_num).float()
            y_test_one_hot = torch.nn.functional.one_hot(torch.tensor(y_test), num_classes=event_num).float()
            if clf_type == 'DDIMDL':
                model = DNN()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters())
                train_dataset = TensorDataset(torch.tensor(x_train).float(), y_train_one_hot)
                test_dataset = TensorDataset(torch.tensor(x_test).float(), y_test_one_hot)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                pred += train_model(model, train_loader, test_loader, NUM_EPOCHS, criterion, optimizer, device)
                continue
            elif clf_type == 'CNN_DDI':
                x_train_reshaped = x_train.reshape(-1, 572, 2)
                x_test_reshaped = x_test.reshape(-1, 572, 2)
                model = CNN_DDI()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters())

                print(x_train_reshaped.shape)
                print(x_test_reshaped.shape)
                print(y_train_one_hot.shape, y_test_one_hot.shape)
                print(model)
                train_dataset = TensorDataset(torch.tensor(x_train_reshaped).float(), y_train_one_hot)
                test_dataset = TensorDataset(torch.tensor(x_test_reshaped).float(), y_test_one_hot)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                pred += train_model(model, train_loader, test_loader, NUM_EPOCHS, criterion, optimizer, device)
                continue
            elif clf_type == 'RF':
                clf = RandomForestClassifier(n_estimators=100)
            elif clf_type == 'GBDT':
                clf = GradientBoostingClassifier()
            elif clf_type == 'SVM':
                clf = SVC(probability=True)
            elif clf_type == 'FM':
                clf = GradientBoostingClassifier()
            elif clf_type == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=4)
            else:
                clf = LogisticRegression()
            clf.fit(x_train, y_train)
            pred += clf.predict_proba(x_test)
        pred_score = pred / len(feature_matrix)
        pred_type = np.argmax(pred_score, axis=1)
        y_true = np.hstack((y_true, y_test))
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))
    result_all, result_eve = evaluate(y_pred, y_score, y_true, event_num, set_name)
    # =============================================================================
    #         a,b=evaluate(pred_type,pred_score,y_test,event_num)
    #         for i in range(all_eval_type):
    #             result_all[i]+=a[i]
    #         for i in range(each_eval_type):
    #             result_eve[:,i]+=b[:,i]
    #     result_all=result_all/5
    #     result_eve=result_eve/5
    # =============================================================================
    return result_all, result_eve


def evaluate(pred_type, pred_score, y_test, event_num, set_name):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))

    precision, recall, th = multiclass_precision_recall_curve(y_one_hot, pred_score)

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
    return [result_all, result_eve]


def self_metric_calculate(y_true, pred_type):
    y_true = y_true.ravel()
    y_pred = pred_type.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_pred_c = y_pred.take([0], axis=1).ravel()
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(len(y_true_c)):
        if (y_true_c[i] == 1) and (y_pred_c[i] == 1):
            TP += 1
        if (y_true_c[i] == 1) and (y_pred_c[i] == 0):
            FN += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 1):
            FP += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 0):
            TN += 1
    print("TP=", TP, "FN=", FN, "FP=", FP, "TN=", TN)
    return (TP / (TP + FP), TP / (TP + FN))


def multiclass_precision_recall_curve(y_true, y_score):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_score_c = y_score.take([0], axis=1).ravel()
    precision, recall, pr_thresholds = precision_recall_curve(y_true_c, y_score_c)
    return (precision, recall, pr_thresholds)


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def drawing(d_result, contrast_list, info_list):
    column = []
    for i in contrast_list:
        column.append(i)
    df = pd.DataFrame(columns=column)
    if info_list[-1] == 'aupr':
        for i in contrast_list:
            df[i] = d_result[i][:, 1]
    else:
        for i in contrast_list:
            df[i] = d_result[i][:, 2]
    df = df.astype('float')
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    df.plot.box(ylim=[0, 1.0], grid=True, color=color)
    return 0


def save_result(feature_name, result_type, clf_type, result):
    with open(feature_name + '_' + result_type + '_' + clf_type+ '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

# Instead of using argparse for command-line arguments, set the parameters directly.
# args = {
#     "featureList": ["smile", "target", "enzyme", "category"],
#     "classifier": ["DDIMDL"],
#     "NLPProcess": "read"
# }

# Main function adjusted for Jupyter Notebook
def main(args):
    seed = 0
    # CV = 5
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
    result_all = {}
    result_eve = {}
    all_matrix = []
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

    for feature in feature_list:
        print(feature)
        new_feature, new_label, event_num = prepare(df_drug, [feature], vector_size, mechanism, action, drugA, drugB)
        all_matrix.append(new_feature)

    start = time.perf_counter()

    for clf in clf_list:
        print(clf)
        all_result, each_result = cross_validation(all_matrix, new_label, clf, event_num, seed, CV, set_name)
        save_result(featureName, 'all', clf, all_result)
        save_result(featureName, 'each', clf, each_result)
        result_all[clf] = all_result
        result_eve[clf] = each_result
    print("time used:", time.perf_counter() - start)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f","--featureList",default=["smile","target","enzyme", "category"],help="features to use",nargs="+")
    parser.add_argument("-c","--classifier",choices=["CNN_DDI","DDIMDL","RF","KNN","LR"],default=["DDIMDL"],help="classifiers to use",nargs="+")
    parser.add_argument("-p","--NLPProcess",choices=["read","process"],default="read",help="Read the NLP extraction result directly or process the events again")
    args=vars(parser.parse_args())
    print(args)
    main(args)
    print('done')
