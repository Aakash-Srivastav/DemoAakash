import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, confusion_matrix,precision_score, recall_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

data = pd.read_excel('SalesDB.xlsx')

data['Total_Sales'] = data['B_Product_Unit_Price'] * data['M_Quantity']
data['Total_Cost'] = (data['B_Product_Unit_Cost'] * data['M_Quantity']) + data['M_Shipping_Cost'] + ((data['M_Discount_%']/100) * data['Total_Sales'])
data['Profit'] = data['Total_Sales'] - data['Total_Cost']
data['Profit_Margin'] = data['Profit']/data['Total_Sales']
data['Year'] = pd.DatetimeIndex(data['A_Date']).year
data['Month'] = pd.DatetimeIndex(data['A_Date']).month
data['B_Product_Category'] = LabelEncoder().fit_transform(data['B_Product_Category'])
data['D_Store_Location'] = LabelEncoder().fit_transform(data['D_Store_Location'])
data['C_Customer_Region'] = LabelEncoder().fit_transform(data['C_Customer_Region'])
data['Order_Type'] = LabelEncoder().fit_transform(data['Order_Type'])
data['Payment_Method'] = LabelEncoder().fit_transform(data['Payment_Method'])

#EDA
print(data.info())
print(data.isnull().sum())
print(data.describe(include='all'))

numeric_cols = ['B_Product_Unit_Cost','B_Product_Unit_Price','M_Shipping_Cost','Total_Sales','Total_Cost','Profit','Profit_Margin']

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

corr_matrix = data.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#Model Training
X = data.drop(columns=["Transaction_ID","A_Date_ID","A_Date_ID","A_Date","A_Date_Day_of_Week","B_Product_ID","B_Product_Name", "C_Customer_ID","C_Customer_Name","C_Customer_Join_Date","D_Store_ID","D_Store_Name","Profit_Margin"], axis=1, inplace=False)

y = data[["Profit_Margin"]]

# Split into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

def reg_eval(model, y_pred):
    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score for {model} : ", r2)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE for {model} : ", mse)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE for {model} : ", mae)

def imp_feature(model):

    # Create a DataFrame with features and their importance scores
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Display the table
    print(feature_importances)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance from Random Forest')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def linear_regression():

    model_name = "Linear Regression"
    # Fit Linear Regression Model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predict on test set
    y_pred = model.predict(x_test)

    reg_eval(model=model_name, y_pred=y_pred)

def random_forest_reg():

    model_name = "Random Forest Regression"
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(x_train, y_train)

    # Predict on test set
    y_pred = model.predict(x_test)

    reg_eval(model=model_name, y_pred=y_pred)

    imp_feature(model=model)

linear_regression()
random_forest_reg()

data['High_Profit'] = [1 if value >= 0.2 else 0 for value in data['Profit_Margin']]

X_new = data.drop(columns=["Transaction_ID","A_Date_ID","A_Date_ID","A_Date","A_Date_Day_of_Week","B_Product_ID","B_Product_Name", "C_Customer_ID","C_Customer_Name","C_Customer_Join_Date","D_Store_ID","D_Store_Name","Profit_Margin","High_Profit"], axis=1, inplace=False)

y_new = data[["High_Profit"]]

# Split into training and testing
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=10)

def class_eval(model, y_pred):

    #Accuracy
    accuracy = accuracy_score(y_test_new, y_pred)
    print(f"Accuracy for {model} : ", accuracy)

    #Precision
    precision = precision_score(y_test_new, y_pred)
    print(f"Precision for {model} : ", precision)

    #Recall
    recall = recall_score(y_test_new, y_pred)
    print(f"Recall for {model} : ", recall)

    #F1-Score
    f1 = f1_score(y_test_new, y_pred)
    print(f"F1-Score for {model} : ", f1)

    #Confusion Matrix
    cm = confusion_matrix(y_test_new, y_pred)
    print(f"Confusion Matrix for {model} : \n", cm)

def log_reg():
    model_name = "Logistic Regression"

    model = LogisticRegression()
    model.fit(x_train_new, y_train_new)

    # Predict on test set
    y_pred_new = model.predict(x_test_new)

    class_eval(model=model_name, y_pred=y_pred_new)

def random_forest_class():

    model_name = "Random Forest Classification"
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(x_train_new, y_train_new)

    # Predict on test  set
    y_pred_new = model.predict(x_test_new)

    class_eval(model=model_name, y_pred=y_pred_new)

    imp_feature(model=model)

log_reg()

random_forest_class()
