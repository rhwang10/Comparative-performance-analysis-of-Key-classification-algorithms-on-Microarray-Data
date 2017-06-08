from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import svm, metrics, grid_search
from sklearn.metrics import confusion_matrix, recall_score, classification_report
import matplotlib.pyplot as plt


from parse import *
import numpy as np
import math

def main():

    try:
        colon_file = open('/home/rhwang1/cs68/labs/Project-rhwang1-msong2/code/data/colon.csv', 'r')  #Open dataset file
        colon_labels_file = open('/home/rhwang1/cs68/labs/Project-rhwang1-msong2/code/data/colon_labels.csv', 'r')

        mosq_file = open('/home/rhwang1/cs68/labs/Project-rhwang1-msong2/code/data/mosquito.csv', "r")  #Open dataset file
        mosq_labels_file = open('/home/rhwang1/cs68/labs/Project-rhwang1-msong2/code/data/mosquito_labels.csv', 'r')

        breast_file = open('/scratch/msong2/formattedData.csv', 'r')
    except:
        print "File doesn't exist!"


    # Parses colon cancer dataset
    colon_data = create_colonData(colon_file)
    colon_data = np.array(colon_data).astype(float) #Each list represents 1 tissue
    colon_labels = parse_colonLabels(colon_labels_file)
    colon_labels = np.array(colon_labels).astype(int)

    # Parses mosquito data and labels into numpy arrays
    # Data is floats
    mosquito_data = parse_mosquitoData(mosq_file)
    mosquito_labels = parse_mosquitoLabels(mosq_labels_file)

    # Parses Breast Cancer dataset
    breast_data, breast_labels = parse_breastData(breast_file)

    #Testing Colon Data
    test_rf(colon_data, colon_labels, "colon")
    test_knn(colon_data, colon_labels, "colon")
    test_svm(colon_data, colon_labels, "colon")

    #Testing Mosquito data
    test_rf(mosquito_data, mosquito_labels, "mosquito")
    test_knn(mosquito_data, mosquito_labels, "mosquito")
    test_svm(mosquito_data, mosquito_labels, "mosquito")

    #Testing Breast cancer data
    test_rf(breast_data, breast_labels, "breast")
    test_knn(breast_data, breast_labels, "breast")
    test_svm(breast_data, breast_labels, "breast")

'''
Tests support vector machine classifier
'''
def test_svm(data, labels, name):
    if name == "colon":
        avg = 'binary'
    else:
        avg = 'micro'
    svm_parameters = [{'C' : [1,10,100,1000], 'kernel': ['linear']}, {'C': [1, 10, 100, 1000], 'gamma': [0, 0.001, 0.0001], 'kernel': ['rbf']}]


    # Create classifier, tune using exhaustive grid search, and fit data
    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, svm_parameters)

    # Print accuracy using cross_val_score metric
    scores = cross_validation.cross_val_score(clf, data, labels, cv=5)
    print("Accuracy of svm on %s: %0.2f (+/- %0.2f)" % ((name), scores.mean(), scores.std() * 2))

    # Print Confusion Matrix metric
    clf_labels = cross_validation.cross_val_predict(clf, data, labels, cv=5)
    unique_labels = np.unique(labels)
    cm = confusion_matrix(labels, clf_labels, labels=unique_labels)
    report = classification_report(labels, clf_labels)
    print "Confusion Matrix for svm on %s\n" % (name)
    print cm
    print "Classification Report:\n"
    print report
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm, unique_labels)
    plt.show()

'''
Tests K-nearest neighbors
'''
def test_knn(data, labels, name):
    if name == "colon":
        avg = 'binary'
    else:
        avg = 'micro'
    knn_parameters = {'n_neighbors': [1,10], 'weights': ['uniform', 'distance']}


    # Create classifier, tune using exhaustive grid search, and fit data
    neighbors = KNeighborsClassifier()
    neigh = grid_search.GridSearchCV(neighbors, knn_parameters)

    # Print accuracy using cross_val_score metric
    scores = cross_validation.cross_val_score(neigh, data, labels, cv=5)
    print "Accuracy of knn on %s: %0.2f (+/- %0.2f)" % ((name), scores.mean(), scores.std() * 2)

    # Print Confusion Matrix Metric
    clf_labels = cross_validation.cross_val_predict(neigh, data, labels, cv=5)
    unique_labels = np.unique(labels)
    cm = confusion_matrix(labels, clf_labels, labels=unique_labels)
    report = classification_report(labels, clf_labels)
    print "Confusion Matrix for knn on %s\n" % (name)
    print cm
    print "Classification Report:\n"
    print report
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm, unique_labels)
    plt.show()


"""
Tests random forest classifier
"""
def test_rf(data, labels, name):
    if name == "colon":
        avg = 'binary'
    else:
        avg = 'micro'
    rf_parameters = [{'n_estimators': [10, 100], 'max_features':['sqrt', 'log2']}]

    # Create classifier, tune using exhaustive grid search, and fit data
    rf = RandomForestClassifier()
    random_forest = grid_search.GridSearchCV(rf, rf_parameters)

    # Print accuracy using cross_val_score metric
    scores = cross_validation.cross_val_score(random_forest, data, labels, cv=5)
    print "Accuracy of rf on %s: %0.2f (+/- %0.2f)" % ((name), scores.mean(), (scores.std() * 2))

    # Print Confusion Matrix Metric
    clf_labels = cross_validation.cross_val_predict(random_forest, data, labels, cv=5)
    unique_labels = np.unique(labels)
    cm = confusion_matrix(labels, clf_labels, labels=unique_labels)
    report = classification_report(labels, clf_labels)
    print "Confusion Matrix for rf on %s\n" % (name)
    print cm
    print "Classification Report:\n"
    print report
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm, unique_labels)
    plt.show()


'''
Generates confusion matrix plot_confusion_matrix
'''

def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


main()
