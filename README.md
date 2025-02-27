# Air-pollution-prediction using machine learning
# Aim
The primary aim of an AI-powered Air Pollution Prediction project using Machine Learning is to develop a model that can accurately forecast air quality levels based on real-time and historical data. This helps in mitigating health risks, guiding environmental policies, and raising awareness about air pollution trends.

## About
Project Overview
This project aims to develop a machine learning model to predict air pollution levels using real-time and historical data. By analyzing key air pollutants like PM2.5, PM10, NO2, SO2, CO, and O3, the system forecasts the Air Quality Index (AQI) to provide timely warnings and insights. The project also includes real-time data analysis and visualization to help users understand pollution trends and take necessary precautions.

## features
## Key Features of the AI-Powered Air Pollution Prediction System
1.Data Collection & Processing

Real-time data fetching from APIs (OpenAQ, AQICN, etc.)
Historical air pollution data integration
Inclusion of meteorological factors (temperature, humidity, wind speed)
Data cleaning, handling missing values, and outlier removal
Machine Learning Models for Prediction

2.Regression models: Linear Regression, Decision Tree Regression
Ensemble models: Random Forest, XGBoost
Deep learning models: LSTMs for time-series forecasting
Model evaluation using RMSE, MAE, and R² Score
Real-Time AQI Prediction & Alerts

3.Live AQI forecasting based on real-time input
Pollution level alerts when air quality exceeds safe limits
Location-based predictions for customized AQI insights
Data Visualization & Insights

4.Interactive graphs and charts for pollution trends
Time-series analysis of air pollutants
Heatmaps and comparative analysis across locations
Deployment & UI

5.Flask/Django backend with REST API support
User-friendly web interface using HTML, CSS, JavaScript
Cloud deployment on AWS/GCP/Heroku for scalability
## code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
%matplotlib inline
sns.set()
df=pd.read_csv("air_quality_india.csv")
df
df.head()
df.tail()
df.shape
df.info()
df.isnull().sum()
df.describe().T.style.background_gradient(cmap="Blues")
print(df["PM2.5"].describe())
df.nunique()
pd.DataFrame(df["Year"].value_counts())
pd.DataFrame(df["Month"].value_counts().sort_index(ascending=True))
pd.DataFrame(df["Hour"].value_counts().sort_values(ascending=False))
pd.DataFrame(df["PM2.5"].sort_values(ascending=False).head(15))

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract new features
df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2]
                                                 else 'Summer' if x in [3, 4, 5]
                                                 else 'Monsoon' if x in [6, 7, 8]
                                                 else 'Autumn')

print(df.head())  # Check the new features
plt.figure(figsize=(10,6), facecolor="orange", edgecolor="red")
plt.plot(df['Timestamp'], df['PM2.5'], color='blue')  # Change color to blue
plt.title('PM2.5 Levels Over Time')
plt.xlabel('Time')
plt.ylabel('PM2.5 Level')
plt.show()


plt.figure(figsize=(8,6))
sns.histplot(df['PM2.5'], bins=50, kde=True, color='blue')  # Change color to green
plt.title('Distribution of PM2.5 Levels')
plt.xlabel('PM2.5 Level')
plt.show()
# Box plot by season
plt.figure(figsize=(8,6))
sns.boxplot(x='Season', y='PM2.5', data=df)
plt.title('PM2.5 Levels by Season')
plt.show()


# Average PM2.5 by hour
hourly_avg = df.groupby('Hour')['PM2.5'].mean()
plt.figure(figsize=(10,6))
hourly_avg.plot()
plt.title('Average PM2.5 Levels by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('PM2.5 Level')
plt.show()
     #count plot to season

sns.countplot(data =df ,x = "Season")
plt.title('count the season')
plt.xlabel('Season')
plt.ylabel('Count')
plt.show()

sns.set(rc={"figure.figsize": (20,5)})
sns.countplot(data=df, x="Month", palette='Set2')  # Use the 'Set2' color palette for different colors
plt.title('Count of Records for Each Month')
plt.xlabel('Month')
plt.show()

#This is a bar plot that displays the sum of PM2.5 levels for each month.
air = df.groupby(["Month"], as_index = False)["PM2.5"].sum().sort_values(by="PM2.5",ascending = False).head(10)

sns.set(rc={"figure.figsize":(20,5)})
sns.barplot(data = air,x = "Month",y = "PM2.5" )

air = df.groupby(["Day_of_Week"], as_index=False)["PM2.5"].sum().sort_values(by="PM2.5", ascending=False).head(10)

sns.set(rc={"figure.figsize": (20,5)})
sns.barplot(data=air, x="Day_of_Week", y="PM2.5", palette='viridis')  # Use a color palette for different colors
plt.title('PM2.5 Levels by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Total PM2.5')
plt.show()
sns.boxplot(df["PM2.5"])
 Q 1 = df['PM2.5'].quantile(0.25)
Q3 = df['PM2.5'].quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df[~((df['PM2.5'] < (Q1 - 1.5 * IQR)) | (df['PM2.5'] > (Q3 + 1.5 * IQR)))]


df_no_outliers.head(2)
     upper_limit = df['PM2.5'].quantile(0.95)
lower_limit = df['PM2.5'].quantile(0.05)

df['PM2.5'] = np.where(df['PM2.5'] > upper_limit, upper_limit, df['PM2.5'])
df['PM2.5'] = np.where(df['PM2.5'] < lower_limit, lower_limit, df['PM2.5'])
df.head(8)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Season"] = le.fit_transform(df['Season'])
df

plt.figure(figsize=(16, 8))
sns.heatmap(df.corr(), cmap='RdYlBu', annot=True, fmt=".1f", linewidths=.5)  # Professional 'RdYlBu' color palette
plt.show()

X = df.drop(["PM2.5"],axis=1)
y = df["PM2.5"]
X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, shuffle=True)


X_train



pd.DataFrame(y_test)
pd.DataFrame(y_train)
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
     def all(model):
    model.fit(X_train,y_train.astype(int))
    y_pred = model.predict(X_test)
    print("score_test=", model.score(X_test ,y_test.astype(int))*100)
    print("score_train=", model.score(X_train ,y_train.astype(int))*100)
    print("mean_absolute_error=", metrics.mean_absolute_error(y_test, y_pred))
     
In [ ]:


     
In [59]:

# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Load the dataset
file_path = "air_quality_india.csv"  # Ensure file is in the correct location
df = pd.read_csv(file_path)

# Step 3: Handle missing values (if any)
df = df.dropna()  # Remove rows with missing values

# Step 4: Define Features (X) and Target Variable (y)
X = df.drop(columns=["PM2.5", "Timestamp"])  # Features (excluding PM2.5 and Timestamp)
y = df["PM2.5"]  # Target variable

# Step 5: Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Initialize models
models = {
    "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
}

# Step 7: Train each model and evaluate
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)  # Train model
    y_train_pred = model.predict(X_train)  # Predictions on training data
    y_test_pred = model.predict(X_test)  # Predictions on testing data

    # Calculate metrics
    train_score = r2_score(y_train, y_train_pred) * 100  # Convert to percentage
    test_score = r2_score(y_test, y_test_pred) * 100  # Convert to percentage
    mae = mean_absolute_error(y_test, y_test_pred)  # Mean Absolute Error

    # Store results
    results[name] = {
        "Train R² Score": f"{train_score:.2f}%",
        "Test R² Score": f"{test_score:.2f}%",
        "Mean Absolute Error": f"{mae:.4f}"
    }

# Step 8: Print results
for model, metrics in results.items():
    print(f"\nModel: {model}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error,r2_score
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

models = {
    'lr': LinearRegression(),
    'GbR': GradientBoostingRegressor(),
    'KNN': KNeighborsRegressor(n_neighbors=8,algorithm="kd_tree", leaf_size=40),
    'Dtr': DecisionTreeRegressor(),
    'Rfr': RandomForestRegressor(n_estimators=200,max_depth=6),
}

for name, md in models.items():
    md.fit(X_train, y_train)
    y_pred = md.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name}: MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}, R2 Score: {r2}")
import matplotlib.pyplot as plt

models = ['LinearRegression', 'GradientBoostgRegressor', 'KNeighborsRegressor', 'DecisionTreeregressor', 'RandomForest']
mae_scores = [15.619237670923445, 6.4497706884182895,5.55847417633651,2.883991020859235,7.030034126734294 ]
mse_scores = [359.62066855024966,73.90256431271001,56.455451094946305,24.430552838720807,84.62326110328198]
rmse_scores = [18.96366706494948,8.596660067299975,7.513684255739411,4.942727267280768,9.199090232369828]
mape_scores = [41.52442871317679,14.772926814073564,12.833660792035392,6.023831992678887,16.27239499195233]
r2_scores = [0.29357418159888415,0.8548284788885752,0.8891009562833874,0.9520095067042189,0.8337691303404486]

fig, ax = plt.subplots(5, 1, figsize=(11, 18),facecolor='white')

# Plot MAE
ax[0].bar(models, mae_scores, color='lightcoral')
ax[0].set_title('Mean Absolute Error (MAE) for Each Model')
ax[0].set_xlabel('Model')
ax[0].set_ylabel('MAE')


# Plot MSE
ax[1].bar(models, mse_scores, color='lightsalmon')
ax[1].set_title('Mean Squared Error (MSE) for Each Model')
ax[1].set_xlabel('Model')
ax[1].set_ylabel('MSE')


# Plot RMSE
ax[2].bar(models, rmse_scores, color='skyblue')
ax[2].set_title('Root Mean Squared Error (RMSE) for Each Model')
ax[2].set_xlabel('Model')
ax[2].set_ylabel('RMSE')


# Plot MAPE
ax[3].bar(models, mape_scores, color='pink')
ax[3].set_title('Mean Absolute Percentage Error (MAPE) for Each Model')
ax[3].set_xlabel('Model')
ax[3].set_ylabel('MAPE')

ax[4].bar(models, r2_scores, color='orange')
ax[4].set_title('R-Sqaured Score for Each Model')
ax[4].set_xlabel('Model')
ax[4].set_ylabel('R-Squared Score')
for a in ax:
    a.set_facecolor('white')  # Set axes background to white
    a.spines['top'].set_color('black')
    a.spines['bottom'].set_color('black')
    a.spines['left'].set_color('black')
    a.spines['right'].set_color('black')
    a.spines['top'].set_linewidth(1.5)
    a.spines['bottom'].set_linewidth(1.5)
    a.spines['left'].set_linewidth(1.5)
    a.spines['right'].set_linewidth(1.5)

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

models = ['LinearRegression', 'GradientBoostgRegressor', 'KNeighborsRegressor', 'DecisionTreeregressor', 'RandomForest']
mae_scores = [15.619237670923445, 6.4497706884182895,5.55847417633651,2.883991020859235,7.030034126734294 ]
mse_scores = [359.62066855024966,73.90256431271001,56.455451094946305,24.430552838720807,84.62326110328198]
rmse_scores = [18.96366706494948,8.596660067299975,7.513684255739411,4.942727267280768,9.199090232369828]
mape_scores = [41.52442871317679,14.772926814073564,12.833660792035392,6.023831992678887,16.27239499195233]
r2_scores = [0.29357418159888415,0.8548284788885752,0.8891009562833874,0.9520095067042189,0.8337691303404486]

fig, ax = plt.subplots(5, 1, figsize=(9, 12))

# Plot MAE
ax[0].plot(models, mae_scores, marker='o', color='lightcoral', linestyle='-')
ax[0].set_title('Mean Absolute Error (MAE) for Each Model')
ax[0].set_xlabel('Model')
ax[0].set_ylabel('MAE')

# Plot MSE
ax[1].plot(models, mse_scores, marker='o', color='lightsalmon', linestyle='-')
ax[1].set_title('Mean Squared Error (MSE) for Each Model')
ax[1].set_xlabel('Model')
ax[1].set_ylabel('MSE')

# Plot RMSE
ax[2].plot(models, rmse_scores, marker='o', color='blue', linestyle='-')
ax[2].set_title('Root Mean Squared Error (RMSE) for Each Model')
ax[2].set_xlabel('Model')
ax[2].set_ylabel('RMSE')

# Plot MAPE
ax[3].plot(models, mape_scores, marker='o', color='hotpink', linestyle='-')
ax[3].set_title('Mean Absolute Percentage Error (MAPE) for Each Model')
ax[3].set_xlabel('Model')
ax[3].set_ylabel('MAPE')

ax[4].plot(models, r2_scores, marker='o',color='orange')
ax[4].set_title('R-Sqaured Score for Each Model')
ax[4].set_xlabel('Model')
ax[4].set_ylabel('R-Squared Score')

for a in ax:
    a.set_facecolor('white')  # Set axes background to white
    a.spines['top'].set_color('black')
    a.spines['bottom'].set_color('black')
    a.spines['left'].set_color('black')
    a.spines['right'].set_color('black')
    a.spines['top'].set_linewidth(1.5)
    a.spines['bottom'].set_linewidth(1.5)
    a.spines['left'].set_linewidth(1.5)
    a.spines['right'].set_linewidth(1.5)

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

models = ['LinearRegression', 'GradientBoostgRegressor', 'KNeighborsRegressor', 'DecisionTreeregressor', 'RandomForest']
mae_scores = [15.619237670923445, 6.4497706884182895,5.55847417633651,2.883991020859235,7.030034126734294 ]
mse_scores = [359.62066855024966,73.90256431271001,56.455451094946305,24.430552838720807,84.62326110328198]
rmse_scores = [18.96366706494948,8.596660067299975,7.513684255739411,4.942727267280768,9.199090232369828]
mape_scores = [41.52442871317679,14.772926814073564,12.833660792035392,6.023831992678887,16.27239499195233]
r2_scores = [0.29357418159888415,0.8548284788885752,0.8891009562833874,0.9520095067042189,0.8337691303404486]

fig, ax = plt.subplots(5, 1, figsize=(9, 12))

# Plot MAE
ax[0].plot(models, mae_scores, marker='o', color='lightcoral', linestyle='-')
ax[0].set_title('Mean Absolute Error (MAE) for Each Model')
ax[0].set_xlabel('Model')
ax[0].set_ylabel('MAE')

# Plot MSE
ax[1].plot(models, mse_scores, marker='o', color='lightsalmon', linestyle='-')
ax[1].set_title('Mean Squared Error (MSE) for Each Model')
ax[1].set_xlabel('Model')
ax[1].set_ylabel('MSE')

# Plot RMSE
ax[2].plot(models, rmse_scores, marker='o', color='blue', linestyle='-')
ax[2].set_title('Root Mean Squared Error (RMSE) for Each Model')
ax[2].set_xlabel('Model')
ax[2].set_ylabel('RMSE')

# Plot MAPE
ax[3].plot(models, mape_scores, marker='o', color='hotpink', linestyle='-')
ax[3].set_title('Mean Absolute Percentage Error (MAPE) for Each Model')
ax[3].set_xlabel('Model')
ax[3].set_ylabel('MAPE')

ax[4].plot(models, r2_scores, marker='o',color='orange')
ax[4].set_title('R-Sqaured Score for Each Model')
ax[4].set_xlabel('Model')
ax[4].set_ylabel('R-Squared Score')

for a in ax:
    a.set_facecolor('white')  # Set axes background to white
    a.spines['top'].set_color('black')
    a.spines['bottom'].set_color('black')
    a.spines['left'].set_color('black')
    a.spines['right'].set_color('black')
    a.spines['top'].set_linewidth(1.5)
    a.spines['bottom'].set_linewidth(1.5)
    a.spines['left'].set_linewidth(1.5)
 a.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.show()
dtr = DecisionTreeRegressor()
dtr.fit(X_train,y_train)
dtr.predict(X_test)
import pickle
pickle.dump(dtr,open('dtr.pkl','wb'))
import sklearn
print(sklearn.__version__)
     
     


## Architecture Diagram

![arc](https://github.com/user-attachments/assets/8baa98fb-b143-4ea4-8a1e-2d1c80652c13)

## Output:

![o 1](https://github.com/user-attachments/assets/1a970a40-88da-4460-8235-015e02cc3701)




![o2](https://github.com/user-attachments/assets/fe55f56f-5707-4b48-bdd6-db3788077df3)



![o3](https://github.com/user-attachments/assets/58334139-b086-4dd1-93c0-433c271031a8)


![04](https://github.com/user-attachments/assets/98031c92-5cb0-4091-bcea-76aba3b81cd0)



## Results
The AI-powered air pollution prediction project successfully forecasts AQI using real-time and historical data with high accuracy. It provides real-time monitoring, location-based predictions, and alerts for hazardous pollution levels. The system includes interactive visualizations like time-series trends and heatmaps, aiding decision-making for environmental policies. With a user-friendly web interface and scalable cloud deployment, it ensures accessibility for the public and organizations to track and mitigate air pollution effectively.










