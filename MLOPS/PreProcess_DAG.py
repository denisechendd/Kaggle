'''MLOps Pipeline using Apache Airflow: Overview
The given dataset contains app usage behaviour with five key columns:

Date (usage day)
App (e.g., Instagram, WhatsApp)
Usage (minutes spent)
Notifications (alerts received)
and Times Opened (app launches)'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
# load the dataset
data = pd.read_csv('screentime_analysis.csv')

# check for missing values and duplicates
print(data.isnull().sum())
print(data.duplicated().sum())

# convert Date column to datetime and extract features
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month

# encode the categorical 'App' column using one-hot encoding
data = pd.get_dummies(data, columns=['App'], drop_first=True)

# scale numerical features using MinMaxScaler
scaler = MinMaxScaler()
data[['Notifications', 'Times Opened']] = scaler.fit_transform(data[['Notifications', 'Times Opened']])

# feature engineering
data['Previous_Day_Usage'] = data['Usage (minutes)'].shift(1)
data['Notifications_x_TimesOpened'] = data['Notifications'] * data['Times Opened']
# split data into features and target variable
X = data.drop(columns=['Usage (minutes)', 'Date'])
y = data['Usage (minutes)']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')
# define the data preprocessing function
def preprocess_data():
    file_path = 'screentime_analysis.csv'
    data = pd.read_csv(file_path)

    data['Date'] = pd.to_datetime(data['Date'])
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month

    data = data.drop(columns=['Date'])

    data = pd.get_dummies(data, columns=['App'], drop_first=True)

    scaler = MinMaxScaler()
    data[['Notifications', 'Times Opened']] = scaler.fit_transform(data[['Notifications', 'Times Opened']])

    preprocessed_path = 'preprocessed_screentime_analysis.csv'
    data.to_csv(preprocessed_path, index=False)
    print(f"Preprocessed data saved to {preprocessed_path}")

# define the DAG
dag = DAG(
    dag_id='data_preprocessing',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

# define the task
preprocess_task = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess_data,
    dag=dag,
)