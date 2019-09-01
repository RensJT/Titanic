import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from matplotlib.pyplot import figure, show

### Loads Data
data=pd.read_csv('train.csv') #extracts all data
print(data.columns) #Prints column names to select columns for model features

data_test_competition=pd.read_csv('test.csv')#extracts the competition test data

### Manipulates data to add Title column
def title_finder(df):
    """Finds the title for a DataFrame with a column 'Names' made up of string entries.
    The titles must end with a '.' and must be the first words in the strings to do so."""
    all_names=df.Name.unique()
    # Loops over every name (a string) in array all_names, splits each string and finds the title.
    # Titles end with a "." for all titles in dataset (inspected data), which is used to find the titles.
    titles=[]
    for n in all_names:
        title_found=False
        i=0
        while title_found == False:
            i+=1
            try:
                word=n.split()[i]
            except:
                title_found = True #Not true, but this ends the while loop
            if word[-1] == '.':
                titles.append(word)
                title_found = True
    return titles

def title_rarifier(Titles, threshold):
    """If titles is present below the threshold number of times, it will be converted to 'Rare'."""
    #Get series of value counts of titles
    n_titles=pd.Series(Titles).value_counts()
    #Find common titles and put them in a list
    common_titles=n_titles[n_titles > threshold].index.tolist()
    #Itterate through titles looking for rare titles
    for i,t in enumerate(Titles):
        if t not in common_titles:
            Titles[i] = "Rare"
    return Titles

#Find the titles
titles=title_finder(data)
titles_test_competition=title_finder(data_test_competition)

# Convert rarer titles to "Rare", easier for ML to make use of with so few of these rarer occurances
titles=title_rarifier(titles, 20)
titles_test_competition=title_rarifier(titles_test_competition, 20)

data['Title']=titles
data_test_competition['Title']=titles_test_competition

### Manipulates data to add Floor column
def floor_finder(df):
    """#For a DataFrame with column Cabin, returns list of floor the passanger is staying on.
    """
    cabins=df.Cabin.values #Convert to array
    floor=[]
    for c in cabins:
        try:
            floor.append(c[0])
        except:
            floor.append(c)
    return floor

#Find floors for both data sets
floor=floor_finder(data)
floor_test_competition=floor_finder(data_test_competition)

#Add to the main data and fill nan values. Too many nan values for imputation
data['Floor']=floor
data['Floor']=data.Floor.fillna("None given")
data_test_competition['Floor']=floor_test_competition
data_test_competition['Floor']=data_test_competition.Floor.fillna("None given")


# Simple features added
def additional_features(df):
    female = np.array([sex == 'female' for sex in df.Sex])
    child = np.array([a < 18 for a in df.Age])
    parch = np.array([parch != 0 for parch in df.Parch])
    mrs = np.array([T == "Mrs." for T in df.Title])

    df['Child'] = child
    df['Female'] = female
    df['Mother'] = female * parch * mrs
    df['Family Size'] = df.SibSp + df.Parch
    df['Ticket Frequency'] = df.Ticket.value_counts()[df.Ticket].tolist()
    df['Family Name'] = [name.split(",")[0] for name in df.Name]

    # Survival rate by Family
    fam_data = data.groupby('Family Name')  # We need to use just the "data" DF here because the other
    fam_surv_rate = fam_data.Survived.mean()  # doesn't have the survival rate
    fam_surv_rate_column = []
    for surname in df['Family Name']:
        try:
            fam_surv_rate_column.append(fam_surv_rate[surname])
        except:
            fam_surv_rate_column.append(fam_surv_rate.mean())
    # print(fam_surv_rate_column)
    df['Family Survival Rate'] = fam_surv_rate_column

    ## This doesn't seem to be useful
    # Survival rate by Ticket
    tic_data = data.groupby('Ticket')
    tic_surv_rate = tic_data.Survived.mean()
    tic_surv_rate_column = []
    for tic in df['Ticket']:
        try:
            tic_surv_rate_column.append(tic_surv_rate[tic])
        except:
            tic_surv_rate_column.append(None)
    df['Ticket Survival Rate'] = tic_surv_rate_column
    return df


def age_filler(df):
    age_groupby = df.groupby(['Sex', 'Title', 'Pclass']).Age.mean()
    ages = []
    for i in df.index:
        if np.isnan(df.Age[i]):
            ages.append(age_groupby[df.Sex[i], df.Title[i], df.Pclass[i]])
        else:
            ages.append(df.Age[i])
    df['Age'] = ages
    return df


data = additional_features(data)
data_test_competition = additional_features(data_test_competition)
data_test_competition.Title.at[88] = "Mrs."  # Previously 'Ms.': a "Rare" title for a 3rd class passanger.
# Caused error in age filling
data = age_filler(data)
data_test_competition = age_filler(data_test_competition)

#Plotting features for inspection
fig=figure(figsize=(15,20))
for i,feature in enumerate(['Pclass', 'Female', 'Age',
                            'Family Size' , 'Fare', 'Title',
                            'Child', 'Mother', 'Ticket Frequency',
                            'Embarked', 'Floor']):
    feature_survival=data.groupby(feature).Survived.mean()
    ax=fig.add_subplot(4,3,i+1)
    ax.plot(feature_survival, marker='o', markersize=2, linewidth=0)
    ax.title.set_text(feature)
    ax.set_ylim(-0.05,1.05)
show()

# These plots make of the features make me want to change the Age, Family Size, and Ticket Frequency into categorical
# data as the correllation is not a simple linear one, meaning binning this data into categories might help the ML
# algorithm's accuracy. (I will thus also drop the Child feature.)

def data_categoriser(df):
    age_cats = []
    for age in df.Age:
        if age <= 12:
            age_cats.append("Child")
        elif age <= 18:
            age_cats.append("Teen")
        elif age <= 50:
            age_cats.append("Adult")
        elif age <= 65:
            age_cats.append("Old Adult")
        elif age <= 100:
            age_cats.append("Elder")
        else:
            print("Error found, age:", age)

    famsize_cats = []
    for famsize in df['Family Size']:
        if famsize == 0:
            famsize_cats.append("Single")
        elif famsize <= 3:
            famsize_cats.append("Small Family")
        elif famsize <= 6:
            famsize_cats.append("Medium Family")
        elif famsize <= 15:
            famsize_cats.append("Large Family")
        else:
            print("Error found:", famsize)

    ticsize_cats = []
    for ticsize in df['Ticket Frequency']:
        if ticsize == 0:
            ticsize_cats.append("Single Ticket")
        elif famsize <= 4:
            ticsize_cats.append("Small Ticket")
        elif famsize <= 10:
            ticsize_cats.append("Large Ticket")
        else:
            print("Error found:", ticsize)
    df['Age Category'] = age_cats
    df['Family Size Category'] = famsize_cats
    df['Ticket Frequency Category'] = ticsize_cats
    return df


data = data_categoriser(data)
data_test_competition = data_categoriser(data_test_competition)

###Preprocessing and model fitting
features=['Pclass', 'Female', 'Age Category', 'Family Size Category' , 'Fare', 'Title', 'Mother', 'Ticket Frequency Category', 'Floor', 'Embarked']
X=data[features]
y=data.Survived
X_test_comp=data_test_competition[features]

##Finding numerical and categorical columns of preprocessing
s = (X.dtypes == 'object')
cat_cols=list(s[s].index)
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

## Preprocessing and modeling with a pipeline
#Numerical data preprocessor
num=SimpleImputer(strategy='mean')
#Categorical data preprocessor
cat=Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
     ])

#Preprocesser
pre=ColumnTransformer(transformers=[
    ('num', num, num_cols),
    ('cat', cat, cat_cols)
])

### Itterating over max_leaf_nodes to find best best accuarcy
#Current ranges estaablished from previous grid searches.
# Using these for this scrip for faster run time and higher accuracy
max_nodes=np.arange(5, 16, 1)
n_est=np.arange(65, 96, 1)
mean_accs=np.zeros([len(max_nodes), len(n_est)]) #Array of 0's to which the mean accuracies from crossvalidation will be appended

for j,n in enumerate(n_est):
    for i,m in enumerate(max_nodes):
        ## Defining model
        model=RandomForestClassifier(n_estimators=int(n), max_leaf_nodes=int(m), random_state=42)
        ##Modeling, fitting, and predicting (cross validation) with pipeline
        pipe=Pipeline(steps=[('preprocessor', pre), ('model', model)])
        scores=cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
        mean_accs[i,j]=scores.mean()
        #Returns number of models tested
        if m//5 == m/5:
            print(int(m), "out of", int(max(max_nodes)), "complete")
    print(int(n), "out of", int(max(n_est)), "n_estimators complete")

i_flattened_max=np.argmax(mean_accs)
i_max=i_flattened_max//len(n_est)
j_max=i_flattened_max - (i_flattened_max//len(n_est))*len(n_est)
optimum_max_nodes=max_nodes[i_max]
optimum_n_estimators=n_est[j_max]


print("Best number of maximum leaf nodes:", optimum_max_nodes)
print("Best number of n_estimators:", optimum_n_estimators)
print("Accuracy:", mean_accs[i_max][j_max])

fig=figure()
ax=fig.add_subplot(1,1,1)
for i in range(len(n_est)):
    ax.plot(max_nodes, mean_accs[:,i], label="{}".format(n_est[i]))
    #ax.axvline(15, linestyle='--')
#ax.legend()
show()

### Makes and fits final model
model_final=RandomForestClassifier(n_estimators=int(optimum_n_estimators), max_leaf_nodes=int(optimum_max_nodes), random_state=42)
pipe=Pipeline(steps=[('preprocessor', pre), ('model', model_final)])
pipe.fit(X, y)
y_pred=pipe.predict(X_test_comp)

# Writes submission file
submission=pd.DataFrame({'PassengerId': data_test_competition.PassengerId, 'Survived': y_pred})
submission.to_csv('submission.csv', index=False)