#Importing important libraries and modules
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#% matplotlib inline
import seaborn as sns
plt.rcParams.update({'figure.figsize':(8,5),'figure.dpi':100})
from datetime import datetime
#Importing libraries 
from sklearn.tree import DecisionTreeRegressor
#metrics import
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import warnings    
warnings.filterwarnings('ignore')

#reading the csv file and converting it to pandas dataframes
sales_df = pd.read_csv("Rossmann Stores Data.csv",parse_dates=['Date'])
stores_df = pd.read_csv("store.csv")

def fill_missing_values_store(stores_df):
    stores_df['CompetitionDistance'].fillna(stores_df['CompetitionDistance'].median(), inplace=True)
    stores_df['CompetitionOpenSinceMonth'].fillna(stores_df['CompetitionOpenSinceMonth'].mode()[0], inplace=True)
    stores_df['CompetitionOpenSinceYear'].fillna(stores_df['CompetitionOpenSinceYear'].mode()[0], inplace=True)
    stores_df['Promo2SinceWeek'].fillna(value=0, inplace=True)
    stores_df['Promo2SinceYear'].fillna(value=0, inplace=True)
    stores_df['PromoInterval'].fillna(value=0, inplace=True)
    
def create_date_features(df):
    # Creating features from the date
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfYear'] = df['Date'].dt.dayofyear
    

def transform_data(df1):
    #changing into boolean 
    df1['StateHoliday'].replace({'a':1, 'b':1,'c':1}, inplace=True)
    #combining competition open since month and year into total months
    df1['CompetitionOpen'] = (df1['Year'] - df1['CompetitionOpenSinceYear'])*12 + (df1['Month'] - df1['CompetitionOpenSinceMonth'])
    #correcting the neg values
    df1['CompetitionOpen'] = df1['CompetitionOpen'].apply(lambda x:0 if x < 0 else x)
    #dropping both the columns
    df1.drop(['CompetitionOpenSinceMonth','CompetitionOpenSinceYear'], axis=1,inplace=True)
    #changing promo2 features into meaningful inputs
    #combining promo2 to total months
    df1['Promo2Open'] = (df1['Year'] - df1['Promo2SinceYear'])*12 + (df1['WeekOfYear'] - df1['Promo2SinceWeek'])*0.230137
    #correcting the neg values
    df1['Promo2Open'] = df1['Promo2Open'].apply(lambda x:0 if x < 0 else x)*df1['Promo2']
    #creating a feature for promo interval and checking if promo2 was running in the sale month
    def promo2running(df):
        month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
        try:
            months = df['PromoInterval'].split(',')
            if df['Month'] and month_dict[df['Month']] in months:
                return 1
            else:
                return 0
        except Exception:
            return 0

    #Applying 
    df1['Promo2running'] = df1.apply(promo2running,axis=1)*df1['Promo2']
    #Dropping unecessary columns
    df1.drop(['Promo2SinceYear','Promo2SinceWeek','PromoInterval'],axis=1,inplace=True)
    return df1

def data_split(df2):
    #slicing the most recent six weeks and creating train and test set
    #train
    start_train = pd.to_datetime("2013-01-01")
    end_train = pd.to_datetime("2015-06-14")
    df_train = df2.loc[start_train:end_train]
    #test
    start_test = pd.to_datetime("2015-06-15")
    end_test = pd.to_datetime("2015-07-31")
    df_test = df2.loc[start_test:end_test]

    #csv
    df_train1 = df_train.to_csv("df_train.csv")
    df_test2 = df_test.to_csv("df_test.csv")

    #X and y split for train and test 
    X_train = df_train.drop('Sales',axis=1)
    y_train = df_train[['Sales']]
    X_test = df_test.drop('Sales',axis=1)
    y_test = df_test[['Sales']]
    return X_train, y_train, X_test, y_test

def One_hot_encode(X_train,X_test,categorical_cols):
    #importing
    from sklearn.preprocessing import OneHotEncoder
    #fit encoder
    encoder = OneHotEncoder(sparse_output=False)
    # train
    encoder.fit(X_train[categorical_cols])
    encoded_features = list(encoder.get_feature_names_out(categorical_cols))
    X_train[encoded_features] = encoder.transform(X_train[categorical_cols])
    # test
    X_test[encoded_features] = encoder.transform(X_test[categorical_cols])
    # drop original features
    X_train.drop(categorical_cols,axis=1,inplace=True)
    X_test.drop(categorical_cols,axis=1,inplace=True)
    
from sklearn.preprocessing import StandardScaler
def stand_scale(X_train, X_test, y_train, y_test):
    # Initialize scalers
    stdsc = StandardScaler()
    scaler = StandardScaler()

    # Scaling X_train and X_test
    X_train_scaled = stdsc.fit_transform(X_train)
    X_test_scaled = stdsc.transform(X_test)

    # Scaling y_train and y_test
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

#function to evaluate the model
def model_evaluation(model_name,model_variable,X_train,y_train,X_test,y_test):
    ''' This function predicts and evaluates various models for regression algorithms, visualizes results 
        and creates a dataframe that compares the various models.'''
        
    #making predictions
    y_pred_train = model_variable.predict(X_train)
    y_pred_test = model_variable.predict(X_test)
    
    # Plot the test results
    a = y_test.copy()
    a['Pred Sales'] = y_pred_test.tolist()
    df_plot = a.reset_index(level=['Date'])
    #plot = df_plot.groupby('Date')['Sales','Pred Sales'].sum()
    plot = df_plot.groupby('Date')[['Sales', 'Pred Sales']].sum()
    sns.lineplot(data = plot)
    plt.ylabel("Total Sales and Predicted Sales")
    plt.xticks(rotation = 25)

    #calculate metrics and print the results for test set
    #Mean Absolute Error or MAE
    MAE_train = round(mean_absolute_error(y_train,y_pred_train),6)
    MAE_test = round(mean_absolute_error(y_test,y_pred_test),6)
    #Mean Squared Error or MSE
    MSE_train = round(mean_squared_error(y_train,y_pred_train),6)
    MSE_test = round(mean_squared_error(y_test,y_pred_test),6)
    #Root Mean Squared Error or RMSE
    RMSE_train = round(mean_squared_error(y_train,y_pred_train,squared=False),6)
    RMSE_test = round(mean_squared_error(y_test,y_pred_test,squared=False),6)
    #R2
    R2_train = round(r2_score(y_train, y_pred_train),6)
    R2_test = round(r2_score(y_test, y_pred_test),6)
    #Adjusted R2
    Adj_r2_train = round(1 - (1-r2_score(y_train, y_pred_train)) * (len(y_train)-1)/(len(y_train)-X_train.shape[1]-1),6)
    Adj_r2_test = round(1 - (1-r2_score(y_test, y_pred_test)) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1),6)
    #printing test results
    print(f'The Mean Absolute Error for the validation set is {MAE_test}')
    print(f'The Mean Squared Error for the validation set is {MSE_test}')
    print(f'The Root Mean Squared Error for the validation set is {RMSE_test}')
    print(f'The R^2 for the validation set is {R2_test}')
    print(f'The Adjusted R^2 for the validation set is {Adj_r2_test}')

#    #Saving our results
#    global comparison_columns
#    metric_scores = [model_name,MAE_train,MSE_train,RMSE_train,R2_train,Adj_r2_train,MAE_test,MSE_test,RMSE_test,R2_test,Adj_r2_test]
#    final_dict = dict(zip(comparison_columns,metric_scores))
#    return [final_dict]

fill_missing_values_store(stores_df)

#merge the datasets on stores data
df = sales_df.merge(right=stores_df, on="Store", how="left")

#change into int type
df['StateHoliday'].replace({'0':0}, inplace=True)

# Creating features from the date
create_date_features(df)

#since the stores closed had 0 sale value; removing the irrelevant part
df1 = df[df.Open != 0]
df1.drop('Open', axis=1, inplace=True)

## transformation
df1['Sales'] = np.log(df1['Sales'])

df1.dropna(inplace=True)
df1.drop(df1[df1['Sales'] == float("-inf")].index,inplace=True)

df1 = transform_data(df1)

#setting date and store as index
df1.set_index(['Date','Store'],inplace=True)
#sorting index following the time series
df1.sort_index(inplace=True)

#just in case something messes up
df2 = df1.copy()

#Sales should be the last col
columns=list(df2.columns)
columns.remove('Sales')
columns.append('Sales')
df2=df2[columns]

# we won't need customers for sales forecasting
df2.drop('Customers',axis=1,inplace=True)

X_train, y_train, X_test, y_test = data_split(df2)

#categorical features
categorical_cols = ['DayOfWeek', 'StoreType', 'Assortment']
One_hot_encode(X_train,X_test,categorical_cols)

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = stand_scale(X_train, X_test, y_train, y_test)

# fitting 
random_forest = RandomForestRegressor(n_estimators=50,random_state=42)
random_forest.fit(X_train,y_train)

#model evaluation 
model_evaluation('Random Forest Regressor',random_forest,X_train,y_train,X_test,y_test)

import pickle
# Save the model to a .pkl file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(random_forest, file)
    

