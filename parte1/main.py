# TUTORIAL

from sklearn import datasets,svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])
predict  = clf.predict(digits.data[-1:])

print('Prediction: ',predict)
print('Target: ', digits.target[-1:])

print('Show Image that correspond to the prediction:')
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title('Prediction Image')
plt.show()

print('Show accuracy graphic:')

# Ignoring Warnings
import warnings
warnings.filterwarnings("ignore")

# Graphics libs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sklearn stuf
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# Array of Classiferis
classifiers = [
    KNeighborsClassifier(n_neighbors=1),
    SVC(gamma='auto'),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

# Graphics labels
log_cols = ["Classifier", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

# Split data in half to train and test

X = digits.data;
y = digits.target;

siz = len(X)
mid = len(X)//2

train = X[:mid]
train_labels = y[:mid]
test = X[mid:siz]
test_labels = y[mid:siz]

# Dictionaries to write the results
acc_dict = {}
cls_dict = {}

# Iterate for classifiers and getting accuracy and classification report for each one
for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(train, train_labels)
    train_predictions = clf.predict(test)
    acc = accuracy_score(test_labels, train_predictions)
    csl = classification_report(test_labels, train_predictions)
    acc_dict[name] = acc
    cls_dict[name] = csl

# Make the accuracy graphic
for clf in acc_dict:
	acc_dict[clf] = acc_dict[clf]
	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
	log = log.append(log_entry)
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
plt.tight_layout()
plt.show()

# Make the accuracy table
result = pd.DataFrame({'Classifier': list(acc_dict.keys()), 'Accuracy' : list(acc_dict.values())})
result = result.sort_values(by='Accuracy', ascending=False)
print(result)

# Print the classification report
for clf in cls_dict:
    print(clf,'\n',cls_dict[clf])