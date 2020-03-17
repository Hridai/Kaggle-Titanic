import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    print('eh?')

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890].drop(['Survived'], axis=1), all_data.loc[891:].drop(['Survived'], axis=1)

def onehotencode_col( in_arr, field_name ):
    # Returns df with the field_name removed and appends the one-hot-encoded
    # Write this to allow the field_name to be a list?
    embarked_col = in_arr[field_name].values
    embarked_col = embarked_col.reshape(1,-1).transpose()
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(embarked_col)
    encoder.categories_
    embarked_col = encoder.transform(embarked_col).toarray()
    embarked_col = pd.DataFrame( embarked_col )
    n = in_all[field_name].nunique()
    embarked_col.columns = ['{}_{}'.format(field_name, n) for n in range(1, n + 1)]
    in_arr = in_arr.drop(field_name,axis=1)
    in_arr = pd.concat([in_arr,embarked_col],axis=1)
    return in_arr

# Import files
in_test = pd.read_csv('C:\\GitProjects\\Kaggle Titanic\\test.csv')
in_train = pd.read_csv('C:\\GitProjects\\Kaggle Titanic\\train.csv')
backup_in_test = in_test
train_results = in_train.iloc[:,1].values
in_all = pd.concat([in_train,in_test]).reset_index(drop=True)
del in_train, in_test # We only needed these to concat together for now. Will reinstate split later

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
del df_results

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
### Data Cleanup

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
in_all['Embarked'].fillna('S',inplace=True)
# There is a Fare value missing
in_all.corr()['Fare'].sort_values() # medium negative corr with Pclass
median_fare = in_all.groupby(['Pclass','Parch','SibSp']).median()['Fare'][3][0][0]
in_all['Fare'].fillna(median_fare,inplace=True)

# Cabins require a more nuanced approach
disp = in_all['Cabin'].value_counts()
# Reduce them to just their primary letter
in_all['Cabin'].fillna('M',inplace=True)
in_all['Cabin'].apply(lambda x: type(x)).value_counts()
in_all['Cabin'] = in_all['Cabin'].apply(lambda x: x[0])
# We must investigate the relationship now between these classes and Survival
h = in_all.groupby(['Cabin'])['Survived'].count()
in_all['Cabin'].value_counts()
idx_t = in_all[in_all['Cabin'] == 'T'].index
in_all.loc[idx_t,'Cabin'] = 'A'

sns.catplot(x="Cabin", y="Survived", data=in_all, kind="bar")
#We can combine D and E and F and G. Might as well lump A, B, C together for ease
in_all['Deck'] = in_all['Cabin']
in_all['Deck'] = in_all['Deck'].replace(['A','B','C'],'ABC')
in_all['Deck'] = in_all['Deck'].replace(['D','E'],'DE')
in_all['Deck'] = in_all['Deck'].replace(['F','G'],'FG')
in_all['Deck'].value_counts()
sns.catplot(x="Deck", y="Survived", data=in_all, kind="bar")
sns.catplot(x="Embarked", y="Survived", data=in_all, kind="bar")

# We should have no nulls in any features now
in_all.isnull().sum()

###########################################################
###### Final visualation overview of the remaining features
# Analysing the continuous and the discrete features w.r.t. survival
cont_features = ['Age','Fare']
disc_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']
surv = in_all['Survived'] == 1

fig, axs = plt.subplots(1,2,figsize=(20,20))
for i, feature in enumerate(cont_features):
    sns.distplot(in_all[~surv][feature], label='Not Survived',hist=True,color='Pink',ax=axs[i])
    sns.distplot(in_all[surv][feature], label='Survived',hist=True,color='Green',ax=axs[i])    
    axs[i].tick_params(axis='y', labelsize=20)
    axs[i].legend(loc='upper right', prop={'size' : 20})
fig.show()

fig, axs = plt.subplots(2,3,figsize=(20,20))
for i, feature in enumerate(disc_features, 1):
    plt.subplot(2,3,i)
    sns.countplot(x=feature,data=in_all,hue='Survived')
    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
plt.show()

##########################################
###################### Feature Engineering
in_all = in_all.drop(['Cabin'], axis=1) # Deck was created as a result of this- no longer necessary
corr = in_all.corr()

in_train, in_test = divide_df(in_all)

# Remove the continuous features by creating bins and discretizing them
in_all['Fare'] = pd.qcut(in_all['Fare'], 13)
in_all['Age'] = pd.qcut(in_all['Age'],10)

fig, axs = plt.subplots(2,1,figsize=(20,40))
for i, feature in enumerate(['Age','Fare']):
    sns.countplot(x=feature,hue='Survived',data=in_all,ax=axs[i])
plt.show()

in_all['Family Size'] = in_all['SibSp'] + in_all['Parch'] + 1
in_all['Family Size'] = in_all['Family Size'].replace([1],'Alone')
in_all['Family Size'] = in_all['Family Size'].replace([2,3,4],'Small')
in_all['Family Size'] = in_all['Family Size'].replace([5,6],'Medium')
in_all['Family Size'] = in_all['Family Size'].replace([6,7,8,9,10,11],'Large')
sns.countplot(x='Family Size', hue='Survived', data=in_all)

in_all['Ticket Frequency'] = in_all.groupby('Ticket')['Ticket'].transform('count')
sns.countplot(x='Ticket Frequency',hue='Survived',data=in_all)
in_all = in_all.drop(['Ticket'],axis=1)

in_all['Title'] = in_all['Name'].str.split(', ', expand=True)[1].str.split('.',expand=True)[0]
sns.countplot(x='Title',hue='Survived',data=in_all)
in_all['Title'] = in_all['Title'].replace(['Miss','Mrs','Ms'],'Mrs/Miss/Ms')
in_all['Title'] = in_all['Title'].replace(['the Countess','Lady','Mme','Capt','Sir','Jonkheer','Dona','Don','Mlle','Major','Col','Dr','Rev'],'Other')
in_all['Title'].value_counts()
sns.countplot(x='Title',hue='Survived',data=in_all)
in_all = in_all.drop(['Name'],axis=1)

########################################
######## Encoding data prior to training
# Label Encoding the discrete and categorical variables
backup = in_all
in_all.info()
encode_list = ['Sex','Age','Fare','Embarked','Deck','Family Size','Title']
from sklearn.preprocessing import LabelEncoder
for feature in encode_list:
    in_all[feature] = LabelEncoder().fit_transform(in_all[feature])

# The below's relative size within the dataset is unimportant (non-ordinal). Age and Fare is ordinal.
non_ordinal_list = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family Size']
for item in non_ordinal_list:
    in_all = onehotencode_col(in_all, item)

in_all = in_all.drop(['SibSp','Parch','PassengerId'],axis=1)
in_all = in_all.drop(['PassengerId'],axis=1)

in_train, in_test = divide_df(in_all)

'''
The below is a new approach where we will make our own dyam samples out of the
test set and run many different classifiers on it and see which comes up the
best.

The approach:
Done 1) First learn to run the models on the training set spilt by k folds
Done 2) Engineer some of the features and check how much the results move by
Done 3) Hyperparameter Tuning
Half Done 4) Stackin ExtraTrees and RandomForest does not work. Look at Kaggle for other ideas
5) Instead of accuracy, use F1 score
    
New Approach:
Done 1) Clean up the code
Half Done (lookup the metrics) 2) Understand the second part of the code, annotate it to hell (e.g. the probability methods)
3) Tune hyperparameters, see if it can be done even better
Done but got a worse score 4) Stack/boost and see if the results are even better
5) Instead of accuracy, use the F1??
'''

'''
##########################################################################################
All of the below might be redundant. Look nto this, we night not need this anymore at all.
It MIGHT be necessary for the ensemble/stacking approach
'''

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
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

# Pick a model after eyeballing the above results
choicemodel = RandomForestClassifier()
choicemodel.fit( in_train, train_results )
out = choicemodel.predict( in_test  )
out = pd.concat([backup_in_test['PassengerId'],pd.DataFrame(out)],axis=1)
pd.DataFrame(out).columns = ['PassengerId','Survived']
pd.DataFrame(out).to_csv('C:\\GitProjects\\Kaggle Titanic\\predictions3.csv', header=True, index=False)

'''
Remove up to here? This does not appear to be relevant anymore
##############################################################
'''

'''
look very carefully at this
'''

in_train = StandardScaler().fit_transform(in_train)
in_test = StandardScaler().fit_transform(in_test)

################
##### Gridsearch
from sklearn.model_selection import GridSearchCV
gridmodel = ExtraTreesClassifier()
param_grid = [
        {'n_estimators':[1750,2000], 'criterion':['gini'],'max_features':['auto','sqrt']}
        ]
gridsearch = GridSearchCV(gridmodel, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
gridsearch.fit( in_train, train_results)
print('lol')
gridsearch.best_estimator_
gs_results = gridsearch.cv_results_
gs_feature_importance = gridsearch.best_estimator_.feature_importances_ # gives the relative importance of each feature
''' you can use the above line to filter out the least important characteristics '''


SEED = 42

## Attempt at 'Bagging' using 3 models
# RandomForest, Extremely Random Trees, KNN
rfc_model = RandomForestClassifier(criterion='gini', n_estimators=1750, max_depth=7,min_samples_split=6,min_samples_leaf=6,max_features='auto',oob_score=True,random_state=SEED,n_jobs=-1,verbose=1)
# rfc_model = RandomForestClassifier(criterion='gini', n_estimators=2500, max_depth=7,min_samples_split=5,min_samples_leaf=5,max_features='auto',oob_score=True,random_state=SEED,n_jobs=-1,verbose=1)
etc_model = ExtraTreesClassifier(max_features='sqrt',n_estimators=2000,oob_score=True, bootstrap=True, verbose=1)
knn_model = KNeighborsClassifier(n_neighbors=3)

class_list = [rfc_model,etc_model]

N = 5
n_models = len(class_list)
in_all_nosurv = in_all.drop(['Survived'],axis=1)
probs = pd.DataFrame(np.zeros((len(in_test), N * n_models * 2)), columns=['Fold_{}_Prob_{}'.format(i, j) for i in range(1, N * n_models + 1) for j in range(2)])
importances = pd.DataFrame(np.zeros((in_train.shape[1], N * n_models)), columns=['Fold_{}'.format(i) for i in range(1, N * n_models + 1)], index=in_all_nosurv.columns)
fprs, tprs, scores = [], [], []

skf = StratifiedKFold(n_splits=N, random_state=N * n_models, shuffle=True)

for i, model in enumerate( class_list ):
    oob = 0
    for fold, (trn_idx, val_idx) in enumerate(skf.split(in_train, y), 1):
        print('Fold {}\n'.format(fold))
        
        # Fitting the model
        model.fit(in_train[trn_idx], y[trn_idx])
        
        # Computing Train AUC score
        trn_fpr, trn_tpr, trn_thresholds = roc_curve(y[trn_idx], model.predict_proba(in_train[trn_idx])[:, 1])
        trn_auc_score = auc(trn_fpr, trn_tpr)
        # Computing Validation AUC score
        val_fpr, val_tpr, val_thresholds = roc_curve(y[val_idx], model.predict_proba(in_train[val_idx])[:, 1])
        val_auc_score = auc(val_fpr, val_tpr)  
          
        scores.append((trn_auc_score, val_auc_score))
        fprs.append(val_fpr)
        tprs.append(val_tpr)
        
        # X_test probabilities
        probs.loc[:, 'Fold_{}_Prob_0'.format(fold + ( N * i ))] = model.predict_proba(in_test)[:, 0]
        probs.loc[:, 'Fold_{}_Prob_1'.format(fold + ( N * i ))] = model.predict_proba(in_test)[:, 1]
        # importances.iloc[:, fold - 1] = leaderboard_model.feature_importances_
        
        try:
            oob += model.oob_score_ / N
            print('Model {} Fold {} OOB Score: {}\n'.format(model.__class__, fold + ( N * i ), model.oob_score_))   
        except:
            print('No OOB score available for {}'.format(model.__class__))
    
# probs_list.append(probs.copy())
# for col in probs.columns:
#     probs[col].values[:] = 0
# final_probs = pd.concat(probs_list,axis=1)

print('Average OOB Score: {}'.format(oob))

# Visualise importances
importances['Mean_Importance'] = importances.mean(axis=1)
importances.sort_values(by='Mean_Importance', inplace=True, ascending=False)
plt.figure(figsize=(15, 20))
sns.barplot(x='Mean_Importance', y=importances.index, data=importances)
plt.xlabel('')
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.title('Random Forest Classifier Mean Feature Importance Between Folds', size=15)
plt.show()

def plot_roc_curve(fprs, tprs):
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(15, 15))
    
    # Plotting ROC for each fold and computing AUC scores
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs), 1):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Fold {} (AUC = {:.3f})'.format(i, roc_auc))
        
    # Plotting ROC for random guessing
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')
    
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    # Plotting the mean ROC
    ax.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC = {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc), lw=2, alpha=0.8)
    
    # Plotting the standard deviation around the mean ROC Curve
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='$\pm$ 1 std. dev.')
    
    ax.set_xlabel('False Positive Rate', size=15, labelpad=20)
    ax.set_ylabel('True Positive Rate', size=15, labelpad=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    ax.set_title('ROC Curves of Folds', size=20, y=1.02)
    ax.legend(loc='lower right', prop={'size': 13})
    
    plt.show()

# plot_roc_curve(fprs, tprs)

# Submit Code
## Takes the simple averages of all the 1 and 0 probabilities
## Uses the larger probability to make final prediction
## So for ensembling models, you must extend this probs array with extra model predict_proba values
class_survived = [col for col in probs.columns if col.endswith('Prob_1')]
probs['1'] = probs[class_survived].sum(axis=1) / ( N * n_models )
probs['0'] = probs.drop(columns=class_survived).sum(axis=1) / ( N * n_models )
probs['pred'] = 0
pos = probs[probs['1'] >= 0.5].index
probs.loc[pos, 'pred'] = 1

y_pred = probs['pred'].astype(int)

submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_df['PassengerId'] = backup_in_test['PassengerId']
submission_df['Survived'] = y_pred.values
pd.DataFrame(submission_df).to_csv('C:\\GitProjects\\Kaggle Titanic\\predictions7.csv', header=True, index=False)
'''
end
'''