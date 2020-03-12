import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def onehotencode_col( in_arr, field_name ):
    embarked_col = in_arr[field_name].values
    embarked_col = embarked_col.reshape(1,-1).transpose()
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(embarked_col)
    encoder.categories_
    embarked_col = encoder.transform(embarked_col).toarray()
    embarked_col = pd.DataFrame( embarked_col )
    embarked_col.columns = encoder.get_feature_names()
    in_arr = in_arr.drop(field_name,axis=1)
    in_arr = pd.concat([in_arr,embarked_col],axis=1)
    return in_arr

# Import files
in_test = pd.read_csv('C:\\GitProjects\\Kaggle Titanic\\test.csv')
in_train = pd.read_csv('C:\\GitProjects\\Kaggle Titanic\\train.csv')
backup_in_test = in_test
train_results = in_train.iloc[:,1].values
in_all = pd.concat([in_train,in_test]).reset_index(drop=True)

#########################################
######## Data exploration / visualisation
in_all.info()
in_all.isnull().sum()
in_all.describe()

df_results = pd.DataFrame(train_results)
df_results = df_results[0].value_counts()
df_results.columns = ['Survived']
df_results.plot(kind='bar')
plt.title('Survivors bar chart')
plt.xlabel('0 = Died, 1 = Survived')
plt.ylabel('Percentage')
plt.show()

fig, axs = plt.subplots(1,5,figsize=(20,4))
for i, f in enumerate(["Fare","Age","Pclass","Parch","SibSp"]):
    sns.distplot(in_all[f].dropna(), kde=False, ax=axs[i]).set_title(f)
    axs[i].set(ylabel='# of Passengers')

plt.subtitle('Feature Histograms (Ignoring Missing Values')
plt.show()

sns.heatmap(in_all.corr(), annot=True, cmap='coolwarm')
plt.show()
in_all.corr()['Survived'].sort_values()

fig, axs = plt.subplots(1, 2, figsize=(12,6))
for i, sex in enumerate(["female", "male"]):
    p = in_all[in_all["Sex"] == sex]["Survived"].value_counts(normalize=True).sort_index().to_frame().reset_index()
    sns.barplot(x=["Perished", "Survived"], y="Survived", data=p, hue="index", ax=axs[i], dodge=False)
    axs[i].set_title("Survival Histogram - {:0.1%} Survived ({})".format(p.loc[1,"Survived"], sex))
    axs[i].set_ylabel("Survival Rate")
    axs[i].get_legend().remove()

in_all['Embarked'].value_counts()
in_all[in_all['Sex']=='female']['Embarked'].value_counts()

fig, axs = plt.subplots(1, 3, figsize=(12,6))
for i, emb in enumerate(["S", "C", "Q"]):
    p = in_all[in_all["Embarked"] == emb]["Survived"].value_counts(normalize=True).sort_index().to_frame().reset_index()
    sns.barplot(x=["Perished", "Survived"], y="Survived", data=p, hue="index", ax=axs[i], dodge=False)
    axs[i].set_title("Survival Histogram - {:0.1%} Survived ({})".format(p.loc[1,"Survived"], emb))
    axs[i].set_ylabel("Survival Rate")
    axs[i].get_legend().remove()

in_all['Pclass'].value_counts()
fig, axs = plt.subplots(1, 3, figsize=(12,6))
for i, pcl in enumerate([1, 2, 3]):
    p = in_all[in_all["Pclass"] == pcl]["Survived"].value_counts(normalize=True).sort_index().to_frame().reset_index()
    sns.barplot(x=["Perished", "Survived"], y="Survived", data=p, hue="index", ax=axs[i], dodge=False)
    axs[i].set_title("Survival Histogram - {:0.1%} Survived ({})".format(p.loc[1,"Survived"], pcl))
    axs[i].set_ylabel("Survival Rate")
    axs[i].get_legend().remove()

##Somehow the below just doesnt work with subplots
# fix, axs = plt.subplots(1,3,figsize=(12,6))
# for i, feat in enumerate(['Pclass','Sex','Embarked']):
#     g = sns.catplot(x=feat,y='Survived',data=in_all,kind='bar', ax=axs[i])
#     axs[i].tick_params(labelsize=15)
#     axs[i].set_ylabel('Survival Probability')
#     axs[i].set_xlabel('lol')
#     # axs[i].set_title('Survival Probability by {}'.format(feat))
# fix.show()

g = sns.catplot(x="Parch", y="Survived", data=in_all, kind="bar")
g.despine(left=True)
g.set_ylabels("Survival Probability")
plt.show()

g = sns.kdeplot(in_all["Age"][in_all["Survived"] == 0], label="Perished", shade=True, color=sns.xkcd_rgb["pale red"])
g = sns.kdeplot(in_all["Age"][in_all["Survived"] == 1], label="Survived", shade=True, color=sns.xkcd_rgb["steel blue"])
plt.xlabel("Age")
plt.ylabel("Density")
plt.show()

g = sns.FacetGrid(in_all, col="Survived")
g = g.map(sns.distplot, "Age")

# Note you can add survived to the y axis below and change the kind to Bar to get something useful
g = sns.catplot("Pclass", col="Embarked",  data=in_all, kind="count", palette="muted")
g = g.set_ylabels("# of Passengers")


#####################################
### Data Cleanup, Feature Engineering

# Drop Fields that you don't need
in_train = in_train.drop(['Survived'],axis=1) # A copy of this is made already

# Filling NAs
in_all.isnull().sum()
# Shows that Age and Embarked needs filling in but Cabin definitely needs dropping
in_all.corr()['Age'].sort_values()
# Age is best correlated with Pclass and thusly should be filled in with PClass values
age_by_pclass_sex = in_all.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(in_all['Age'].median()))

# Filling the missing values in Age with the medians of Sex and Pclass groups
in_all['Age'] = in_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
# Googling the name of the famous survivor tells us she boarded on S with her maid
in_all['Embarked'] = in_all['Embarked'].fillna('S',inplace=True)
# There is a Fare value missing
in_all.corr()['Fare'].sort_values() # medium negative corr with Pclass
median_fare = in_all.groupby(['Pclass','Parch','SibSp']).median()['Fare'][3][0][0]
in_all['Fare'].fillna(median_fare,inplace=True)

# Cabins require a more nuanced approach
disp = in_all['Cabin'].value_counts()
# Reduce them to just their primary letter
in_all['Cabin'].fillna('M',inplace=True)
in_all['Cabin'].apply(lambda x: type(x)).value_counts()
in_all['Cabin'] = disp_in_all['Cabin'].apply(lambda x: x[0])
# We must investigate the relationship now between these classes and Survival
h = in_all.groupby(['Cabin'])['Survived'].count()
in_all['Cabin'].value_counts()
idx_t = in_all[in_all['Cabin'] == 'T'].index
in_all.loc[idx_t,'Cabin'] = 'A'
#### Plot graph(s) here
#https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial

in_train = onehotencode_col(in_train, 'Embarked')
in_test = onehotencode_col(in_test, 'Embarked')

# Label Encoding for Gender
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
in_train['Sex'] = le.fit_transform(in_train['Sex'])
in_test['Sex'] = le.fit_transform(in_test['Sex'])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
in_train = standardscaler.fit_transform(in_train)
in_test = standardscaler.fit_transform(in_test)

'''
The below is a new approach where we will make our own dyam samples out of the
test set and run many different classifiers on it and see which comes up the
best.

The approach:
1) First learn to run the models on the training set spilt by k folds
2) Engineer some of the features and check how much the results move by
3) Hyperparameter Tuning
4) Stack or boost the best classifiers and see if the results are even better
5) Instead of accuracy, use F1 score
'''

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

log_cols = ["Classifier", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

X = in_train
y = train_results

acc_dict = {}

for train_index, test_index in sss.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	for clf in classifiers:
		name = clf.__class__.__name__
		clf.fit(X_train, y_train)
		train_predictions = clf.predict(X_test)
		acc = accuracy_score(y_test, train_predictions)
		if name in acc_dict:
			acc_dict[name] += acc
		else:
			acc_dict[name] = acc

for clf in acc_dict:
	acc_dict[clf] = acc_dict[clf] / 10.0
	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
	log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

choicemodel = RandomForestClassifier()
choicemodel.fit( in_train, train_results )
out = choicemodel.predict( in_test  )
out = pd.concat([backup_in_test['PassengerId'],pd.DataFrame(out)],axis=1)
pd.DataFrame(out).columns = ['PassengerId','Survived']
pd.DataFrame(out).to_csv('C:\\GitProjects\\Kaggle Titanic\\predictions2.csv', header=True, index=False)
