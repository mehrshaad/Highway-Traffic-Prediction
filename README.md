# Highway Traffic Prediction

+ **a)** Download Kaggle open dataset from <https://www.kaggle.com/code/> jaymineshkumarpatel/traffic-prediction/data corresponding to highway traffic prediction models and describe the data.
+ **b)** Apply Random Forest (RF) and Support Vector Machine (SVM) regression models on the data to predict the number of vehicles in all junctions for the next coming year.
+ **c)** Analyze the accuracy and extract the results corresponding to different regression error functions as follows: i. MAE
ii. MSE iii. RMSE iv. Huber Loss v. MAPE

## Abstract

Traffic congestion is rising in cities around the world. Contributing factors include expanding urban populations, aging infrastructure, inefficient and uncoordinated traffic signal timing and a lack of real-time data.

The impacts are significant. Traffic data and analytics company INRIX estimates that traffic congestion cost U.S. commuters $305 billion in 2017 due to wasted fuel, lost time and the increased cost of transporting goods through congested areas. Given the physical and financial limitations around building additional roads, cities must use new strategies and technologies to improve traffic conditions.

The aim of this project was about predicting highway traffic through various junctions including dates.

## Dataset descriptions

We have downloaded highway traffic prediction Kaggle open dataset from <https://www.kaggle.com/code/> jaymineshkumarpatel/traffic-prediction/data corresponding to highway traffic prediction models. The data included highway vehicle transportation data of different junctions and dates, including years, months, days, and hours from 2015-11-01 to 2017-06-30.

The sensors on each of these junctions were collecting data at different times, hence you will see traffic data from different time periods. Some of the junctions have provided limited or sparse data requiring thoughtfulness when creating future projections.

## Models implementation

The two regression models SVR and RandomForest have been implemented and trained with the given dataset.

#### Loading Data
<!-- vahid inja nemodar bezan -->
```python
# reading the dataset
df = pd.read_csv("traffic.csv",parse_dates=True, index_col='DateTime')


# extract year from date
df['Year'] = pd.Series(df.index).apply(lambda x: x.year).to_list()

# extract month from date
df['Month'] = pd.Series(df.index).apply(lambda x: x.month).to_list()

# extract day from date
df['Day'] = pd.Series(df.index).apply(lambda x: x.day).to_list()

# extract hour from date
df['Hour'] = pd.Series(df.index).apply(lambda x: x.hour).to_list()
df["Series"] = [1]*len(df)
# iterates through the rows and updates series value of each row with the dayOfYear value
for index, row in df.iterrows():
    row["Series"]=index.dayofyear
```

#### Data Definition

```python
# feature part
X = df[["Junction", 'Year',  'Series' ,'Hour']].values.tolist()
# label part (vehicle counts)
y = sk.utils.validation.column_or_1d(df[["Vehicles"]].values.tolist())
# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43,train_size=0.8)
```

### SVR (Support Vector Regression)

Support Vector Regression is a supervised learning algorithm that is used to predict discrete values. Support Vector Regression uses the same principle as the SVMs. The basic idea behind SVR is to find the best fit line. In SVR, the best fit line is the hyperplane that has the maximum number of points.

#### Model training
```python
svr = make_pipeline(StandardScaler(), SVR())
svr.fit(X_train, y_train)
print(svr.score(X_test, y_test))
# 0.8237075364902094
```


### RandomForest

Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees.

#### Model training
```python
rf = make_pipeline(StandardScaler(), RandomForestRegressor())
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))
# 0.956102660239957
```
