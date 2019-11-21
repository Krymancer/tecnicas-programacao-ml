import sklearn.datasets as ds
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Classifers to test
from sklearn.metrics import accuracy_score, log_loss,  classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


# Load sklearn dataset and converting do a pandas dataframe
cancer = ds.load_breast_cancer()
data = np.c_[cancer.data, cancer.target]
columns = np.append(cancer.feature_names,['target'])
df = pd.DataFrame(data,columns=columns)

# split the test and target / Estratificando
X = df.drop('target', axis=1).astype('float64')
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=df['target'], shuffle=True, test_size=0.2, random_state=42)

# Standtading features
stds = StandardScaler()
x_train_std = stds.fit_transform(x_train)
x_test_std = stds.transform(x_test)

# Array of Classiferis
classifiers = [
    KNeighborsClassifier(n_neighbors=1),
    SVC(gamma='auto'),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=10),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    Perceptron(max_iter=1000, tol=1e-3, alpha=0.0001, penalty='l2')
]

# Graphics labels
log_cols = ["Classifier","Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

# Dictionaries to write the results
acc_dict = {}
cls_dict = {}

# Iterate for classifiers and getting accuracy and classification report for each one
for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(x_train_std, y_train)
    y_pred = clf.predict(x_test_std)
    acc = accuracy_score(y_test, y_pred)
    csl = classification_report(y_test, y_pred)

    acc_dict[name] = acc
    cls_dict[name] = csl

# Make the accuracy table
result_acc = pd.DataFrame({'Classifier' : list(acc_dict.keys()),'Accuracy' : list(acc_dict.values())})
result_acc = result_acc.sort_values(by='Accuracy', ascending=False)
print('\n',result_acc)

# Print the classification report
for clf in cls_dict:
    print('\n',clf,'\n',cls_dict[clf])

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

# function to plot a graph showing the accuracy for train and test
def plot(cf):
    # Plot the test and train results
    mal_train_X = x_train_std[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = x_train_std[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = x_test_std[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = x_test_std[y_test==1]
    ben_test_y = y_test[y_test==1]

    clf = cf

    scores = [clf.score(mal_train_X, mal_train_y), clf.score(ben_train_X, ben_train_y),
                clf.score(mal_test_X, mal_test_y), clf.score(ben_test_X, ben_test_y)]

    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2),
                        ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells\n' + clf.__class__.__name__, alpha=0.8)
    plt.show()

# make a graph to every classfier
for clf in classifiers:
    clf.fit(x_train_std, y_train)
    plot(clf)