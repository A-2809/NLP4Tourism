#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression, r_regression
import lightgbm as lgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import StratifiedKFold 

#training datasets
train_df = pd.read_csv(r'/home/tanmay/Archana/paper1/new_bs.csv')
#print (train_df)

#checking columns for NaN values
d = train_df.isnull().sum(axis=0)
print(d)

train_df["Country"].unique()

# import seaborn as sns
# plt.figure(figsize=(15,15))
# sns.heatmap(train_df.corr(),cmap='Blues',annot=True)

#dropping rows with nan values
#df = train_df.dropna()

# # #replacing with mean values
#df = train_df.fillna(train_df.mean())

# #replacing with median values
#df = train_df.fillna(train_df.median())

# #replacing missing value with (std.dev-/+mean) 
df = train_df.fillna(abs(train_df.mean() + train_df.std()))

df['Total international arrivals'] = (df['Total international arrivals']-df['Total international arrivals'].min()) / (df['Total international arrivals'].max()-df['Total international arrivals'].min())

#spliting the database
x=df.loc[:,['Total tourism employment (direct) as % of total employment', 'GDP per capita','Pollution','Total international expenditure','Goverment Effectiveness', 'Total Domestic Trip', 'Rule of Law', 'Percent of total protected land','Exchange rate','Trade%GDP', 'Women Buisness and Law index',	'Hospital bed'	, 'Regulatory quality']]#, 'Positive', 'Negative', 'Neutral']]
y=df['Total international arrivals']

#normalising
for i in x.columns:
    x[i] = (x[i]-x[i].min()) / (x[i].max()-x[i].min())

# # Preprocessing the data
# # Scaling the Data
# scaler = MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)
# train_data = scaler.fit_transform(x)
# #train_data = scaler.fit_transform(y)

# #Splitting the dataset into 80-20 
coun = df['Country']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,stratify = coun ,random_state=42)

feature_selector_types = ['r_regression', 'mutual_info_regression'] 
def feature_selector(feature_selector_type): 
    if feature_selector_type == 'mutual_info_regression':
        return SelectKBest(score_func = mutual_info_regression)        
    elif feature_selector_type == 'r_regression':
        return SelectKBest(score_func = r_regression)
    
    
# pipeline parameter grid    
parameters=[
    {
        'clf': LinearRegression(),

    },
    {
        'clf': RandomForestRegressor(),
        'clf__n_estimators': [10,50,100],
        'clf__max_depth': [2, 10,20],
        'clf__random_state': [2]
    },
    {   'clf': SVR(),
        'clf__max_iter': [1, 10, 50],
        'clf__C': [0.1, 10, 50],
        'clf__kernel': ['linear', 'sigmoid', 'poly'],
    },
    {
        'clf': DecisionTreeRegressor(),
        'clf__criterion': ['squared_error', 'absolute_error', 'poisson'],
        'clf__max_depth': [2,10,20],
        'clf__splitter' : ['best', 'random'],
    },
    {
        'clf': AdaBoostRegressor(),
        'clf__base_estimator': [DecisionTreeRegressor(criterion='squared_error', max_depth=20)],
        'clf__n_estimators': [10,50,100],
        'clf__learning_rate': [0.1,1,10],
        'clf__loss' : ['linear', 'square', 'exponential'],
        'clf__random_state': [2],
    },
    {
        'clf' : lgb.LGBMRegressor(boosting_type='gbdt',  objective='regression', num_boost_round=2000, learning_rate=0.01, metric='auc'),
        'clf__num_leaves': [10,5,100],
        'clf__reg_alpha': [0.1, 0.5],
        'clf__min_data_in_leaf': [5,10, 50, 100],
        # 'lambda_l1': [0, 1, 1.5],
        # 'lambda_l2': [0, 1]
    }
]

#from sklearn.model_selection import KFold
raw_result = []
results = pd.DataFrame([])
Model = []
Best_Score = []
Remarks = []
mae = []
mape = []
rmse =[]
R2_Score = []
feature_selctor = []

#kfold = StratifiedKFold(n_splits=10)
#skf = KFold(n_splits=10, shuffle=True)

# storing the best hyperparameters for all the combinations
# for model in parameters: 
#     clf = model.pop('clf')
#     print(f"\nStarted {str(clf)}")
#     print("-------------------------------------------")
#     for feature_selctor_type in feature_selector_types:
#         print("\nStarted feature selection "+ str(feature_selctor_type))
#         print("-------------------------------------------")
        
#         predicted_class_labels=[]
#         actual_class_labels=[] 
#         count=0 
#         probs=[]
#         for train_index, test_index in skf.split(x,y):
#             X_train=[]
#             Y_train=[]
#             X_test=[] 
#             Y_test=[]
#             for item in train_index:
#                 X_train.append(x.iloc[item])
#                 Y_train.append(y.iloc[item])
#             for item in test_index:
#                 X_test.append(x.iloc[item])
#                 Y_test.append(y.iloc[item])
#             count+=1                
#             pipeline = Pipeline([("select", feature_selector(feature_selctor_type)),("clf", clf)])
#             print("\nStarted GridSearchCV")
#             print("-------------------------------------------")
#             grid_model = GridSearchCV(pipeline, model, cv=10,verbose=2, scoring='r2', error_score='raise')
#             grid_model.fit(X_train, Y_train)
#             predicted = grid_model.predict(X_test)
#             print("Done")
#             print('MAE: ', metrics.mean_absolute_error(Y_test, predicted))
#             print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predicted)))
#             print("MAPE:", metrics.mean_absolute_percentage_error(Y_test, predicted))
#             print("R2 Score: ",metrics.r2_score(Y_test, predicted))
#             print(f"Training Score: {grid_model.best_score_}")
#             print(f"Parameters: {grid_model.best_params_}") 
#             print(f"Best Regressor: {grid_model.best_estimator_}")
#             Best_Score.append(grid_model.best_score_)
#             Model.append(clf)
#             Remarks.append(str(grid_model.best_params_))
#             mae.append(metrics.mean_absolute_error(Y_test, predicted))
#             mape.append(metrics.mean_absolute_percentage_error(Y_test, predicted))
#             rmse.append(np.sqrt(metrics.mean_squared_error(Y_test, predicted)))
#             R2_Score.append(metrics.r2_score(Y_test, predicted))
#             feature_selctor.append(feature_selctor_type)
        
        
#     raw_result.append({
#             'Model': clf,
#             'Best_Score': grid_model.best_score_,
#             'Best_Params': grid_model.best_params_
#         })

for model in parameters: 
    clf = model.pop('clf')
    print(f"\nStarted {str(clf)}")
    print("-------------------------------------------")
    # for vectorizer_type in vectorizer_types:
    #     print(f"\nStarted Vectorizer {str(vectorizer_type)}")
    #     print("-------------------------------------------")
    for feature_selctor_type in feature_selector_types:
            print("\nStarted feature selection "+ str(feature_selctor_type))
            print("-------------------------------------------")
            #pipeline = Pipeline([("vector", vectorizer(vectorizer_type)),("select", feature_selector(feature_selctor_type)), ("clf", clf)])
            pipeline = Pipeline([("select", feature_selector(feature_selctor_type)),("clf", clf)])
            print("\nStarted GridSearchCV")
            print("-------------------------------------------")
            grid_model = GridSearchCV(pipeline, model, verbose=2, cv=10, scoring='r2', error_score='raise')
            grid_model.fit(x_train, y_train)
            predicted = grid_model.predict(x_test)
            #### Probabilities
            #y_pred_proba = grid_model.predict_proba(x_test)
            #print(y_pred_proba)
            print("Done")
            print('MAE: ', metrics.mean_absolute_error(y_test, predicted))
            print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
            print("MAPE:", metrics.mean_absolute_percentage_error(y_test, predicted))
            print("R2 Score: ",metrics.r2_score(y_test, predicted))
            print(f"Training Score: {grid_model.best_score_}")
            print(f"Parameters: {grid_model.best_params_}") 
            print(f"Best Regressor: {grid_model.best_estimator_}")
            Best_Score.append(grid_model.best_score_)
            Model.append(clf)
            Remarks.append(str(grid_model.best_params_))
            mae.append(metrics.mean_absolute_error(y_test, predicted))
            mape.append(metrics.mean_absolute_percentage_error(y_test, predicted))
            rmse.append(np.sqrt(metrics.mean_squared_error(y_test, predicted)))
            R2_Score.append(metrics.r2_score(y_test, predicted))
            feature_selctor.append(feature_selctor_type)
                
    raw_result.append({
            'Model': clf,
            'Best_Score': grid_model.best_score_,
            'Best_Params': grid_model.best_params_
        })

results['Model'] = Model
results["feature_selctor"] = feature_selctor
results['MAE'] = mae
results['MAPE'] = mape
results['RMSE'] = rmse
results["R2 Score"] = R2_Score 
results['Best_Score'] = Best_Score
results['Remarks'] = Remarks

#print(raw_result)

results.to_csv("/home/tanmay/Archana/trail_mean+std.csv")
