from numpy.fft import fft, fftfreq
from sklearn.preprocessing import normalize
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import itertools
from itertools import chain, combinations
from sklearn.preprocessing import normalize
import pickle
import copy
import pandas as pd
from IPython.display import display, HTML
from scipy.signal import find_peaks
from scipy.stats import skew, entropy, kurtosis
import seaborn as sns
import pyxai
from pyxai import Learning, Explainer, Tools
from util.explain_models import *


l = 5  # Signal length in seconds
ts = 0.00005 # Sample period in seconds
N = int(l/ts) # Number of samples

def scan_freq_ranges(data_fft, data_representation='sign', n_runs=10, n_features_final = 20, step = 5):

    splits = generate_cross_validation_splits(data_fft)

    # Initialize a list to store metrics
    min_accuracies = None
    mean_accuracies = None
    min_f1 = None
    mean_f1 = None
    for i, (train_machines, test_machine) in enumerate(splits):
        train_data, test_data = get_train_test_data(data_fft, train_machines, test_machine)
        #Remove metadata
        n_features_total = len(train_data.columns) - 1    
        accuracies = []
        f1s = []
        for f_range in range(0, n_features_total - n_features_final - 3, step):
            f_range_end = f_range+n_features_final
            X_train = train_data.iloc[:,f_range:f_range_end].to_numpy()
            X_test = test_data.iloc[:,f_range:f_range_end].to_numpy()

            if data_representation in ['diff', 'sign']:
                X_train = np.diff(X_train)
                X_test = np.diff(X_test)
            if data_representation == 'sign':
                X_train = np.sign(X_train)
                X_test = np.sign(X_test)

            y_train = train_data['y'].to_numpy()
            y_test = test_data['y'].to_numpy()
            
            accuracy = 0
            f1 = 0
            #cm = []
            for j in range(n_runs):
                #clf = RidgeClassifier()
                clf = DecisionTreeClassifier(class_weight='balanced', max_features=7, max_depth=3)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                #model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT, seed=None, test_size=0.5, class_weight='balanced')
                # Compute metrics
                accuracy += accuracy_score(y_test, y_pred)
                f1 += f1_score(y_test, y_pred, average='weighted')
                #cm.append(confusion_matrix(y_test, y_pred))
                
            accuracy = accuracy / n_runs
            

            f1 = f1/ n_runs
            accuracies.append(accuracy)
            f1s.append(f1)
            
        plt.figure(figsize=(15, 10))
        plt.grid()
        plt.title(test_machine)
        plt.plot(accuracies, label="accuracy")
        plt.plot(f1s, label="f1")
        plt.legend()
        plt.show()
        max_value = max(accuracies)
        init_value = 10/3
        best_freqs = [(init_value+ init_value*step*index, value) for index, value in enumerate(accuracies) if value >= 0.9*max_value]

        for best_freq in best_freqs:
            print(f"best freq range in general: [{best_freq[0]}, {best_freq[0] + 10*n_features_final/3}], {best_freq[1]}")
        print()
        if min_accuracies is None:
            min_accuracies = accuracies
            mean_accuracies = accuracies
            mean_f1 = f1s
            min_f1 = f1s
        else:
            min_accuracies = [min(a,b) for a, b in zip(accuracies, min_accuracies)]
            mean_accuracies = [a+b for a, b in zip(accuracies, mean_accuracies)]
            mean_f1 = [a+b for a, b in zip(f1s, mean_f1)]    
            min_f1 = [min(a,b) for a, b in zip(f1s, min_f1)]
              

    mean_accuracies = [x / 6 for x in mean_accuracies]
    mean_f1 = [x / 6 for x in mean_f1]
    plt.figure(figsize=(15, 10))
    plt.grid()
    plt.title("Global Accuracy")
    #peaks, _ = find_peaks(min_accuracies)
    #print("peaks", peaks)
    plt.plot(min_accuracies, label="min accuracy")
    plt.plot(mean_accuracies, label="mean accuracy")
    
    plt.legend()
    #plt.plot(min_accuracies[peaks], "x")
    plt.show()
    plt.figure(figsize=(15, 10))
    plt.grid()
    plt.title("Global F1")
    plt.legend()
    plt.plot(min_f1, label="min F1")
    plt.plot(mean_f1, label="mean F1")
    plt.show()

    max_value = max(min_accuracies)
    best_freqs = [(init_value+ init_value*step*index, value) for index, value in enumerate(min_accuracies) if value >= 0.9*max_value]

    for best_freq in best_freqs:
        print(f"best maxmin freq range in general: [{best_freq[0]}, {best_freq[0] + 10*n_features_final/3}], {best_freq[1]}")
    print()
    max_value = max(mean_accuracies)
    best_freqs = [(init_value+ init_value*step*index, value) for index, value in enumerate(mean_accuracies) if value >= 0.9*max_value]

    for best_freq in best_freqs:
        print(f"best maxmin freq range in general: [{best_freq[0]}, {best_freq[0] + 10*n_features_final/3}], {best_freq[1]}")
    print()
    return best_freqs

def column_power_set(df, min_size, max_size):
    # Get the column names
    columns = df.columns

    # Ensure min_size and max_size are within valid range
    if min_size < 1 or max_size > len(columns) or min_size > max_size:
        raise ValueError("Invalid min_size or max_size")

    # Generate all combinations of columns within the specified range
    power_set = chain.from_iterable(combinations(columns, r) for r in range(min_size, max_size + 1))


    # Convert each combination to a DataFrame and store in a list
    power_set_dfs = []
    for combo in power_set:
        try:
            power_set_dfs.append(df[list(combo)])
        except KeyError as e:
            print(f"KeyError: {e} for combination {combo}")

    return power_set_dfs
    

def cross_validation(data_fft, data_representation='raw', n_runs=10, verbose=False):

    splits = generate_cross_validation_splits(data_fft)

    # Initialize a list to store metrics
    metrics = []
    min_accuracies = None
    min_accuracy = 1
    mean_accuracy = 0
    for i, (train_machines, test_machine) in enumerate(splits):
        train_data, test_data = get_train_test_data(data_fft, train_machines, test_machine)
        X_train = train_data.drop(train_data.columns[-2:], axis=1).to_numpy()
        X_test = test_data.drop(test_data.columns[-2:], axis=1).to_numpy()

        if data_representation in ['diff', 'sign']:
            X_train = np.diff(X_train)
            X_test = np.diff(X_test)
        if data_representation == 'sign':
            X_train = np.sign(X_train)
            X_test = np.sign(X_test)


        y_train = train_data['y'].to_numpy()
        y_test = test_data['y'].to_numpy()

        accuracy = 0
        f1 = 0
            #cm = []
        for j in range(n_runs):
            #clf = RidgeClassifier()
            clf = DecisionTreeClassifier(class_weight='balanced', max_features=8, max_depth=3)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
                
                #model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT, seed=None, test_size=0.5, class_weight='balanced')
                # Compute metrics
            accuracy += accuracy_score(y_test, y_pred)
            f1 += f1_score(y_test, y_pred, average='weighted')
            #cm=confusion_matrix(y_test, y_pred)
                
        accuracy = accuracy / n_runs

        if accuracy < min_accuracy:
            min_accuracy = accuracy
        mean_accuracy += accuracy
        f1 = f1/ n_runs
        # Store metrics
        metrics.append({
                'Split': i + 1,
                'Train Machines': train_machines,
                'Test Machine': test_machine,
                'Accuracy': accuracy,
                'F1-Score': f1
                #'Confusion Matrix': cm
        })

        if verbose:
            #Print metrics
            print(f"Split {i+1}:")
            print(f"  Train Machines: {train_machines}")
            print(f"  Test Machine: {test_machine}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            #print(f"  Confusion Matrix:\n{cm}")
            #print()
            #plot_tree(clf)
            #plt.show()
    mean_accuracy = mean_accuracy/6 
    
    if verbose:
        print("min: ", min_accuracy) 
        print("mean: ", mean_accuracy) 

    return min_accuracy, mean_accuracy    

def generate_cross_validation_splits(data, machine_column='Machine'):
    # Get unique machine names
    machine_names = data[machine_column].unique()

    # Generate all combinations of 1 machine for testing
    test_combinations = list(itertools.combinations(machine_names, 1))

    # Generate the corresponding train sets
    train_splits = []
    for test_set in test_combinations:
        train_set = [machine for machine in machine_names if machine not in test_set]
        train_splits.append((train_set, test_set[0]))

    return train_splits

def get_train_test_data(data, train_machines, test_machine, machine_column='Machine'):
    train_data = data[data[machine_column].isin(train_machines)]
    test_data = data[data[machine_column] == test_machine]
    return train_data, test_data

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def sign_of_first_order_difference(data):
    data_diff = data
    data_diff[data_diff.columns[:-1]] = data_diff[data_diff.columns[:-1]].diff(axis = 'columns')
    data_diff = data_diff.drop(data_diff.columns[0], axis=1)
    def transform1(x):
        return False if x < 0 else True
    data_diff[data_diff.columns[:-1]] = data_diff[data_diff.columns[:-1]].map(transform1)
    return data_diff

def first_order_difference(data):
    data_diff = data
    data_diff[data_diff.columns[:-1]] = data_diff[data_diff.columns[:-1]].diff(axis = 'columns')
    data_diff = data_diff.drop(data_diff.columns[0], axis=1)
    return data_diff
    
