# 🛍️ SalesDB Analysis and Profit Prediction

This project analyzes retail transaction data from SalesDB.xlsx, engineering new features such as Total_Sales, Total_Cost, and Profit_Margin. It performs exploratory data analysis (EDA) and builds both regression and classification models to predict profitability.

# 📁 Files

SalesDB.xlsx - Source dataset containing sales transactions

sales_analysis.py - Main Python script for data preprocessing, analysis, modeling, and evaluation

# 📊 Features Created

Total_Sales = Unit Price × Quantity

Total_Cost = (Unit Cost × Quantity) + Shipping Cost + Discount

Profit = Total Sales − Total Cost

Profit_Margin = Profit ÷ Total Sales

Temporal features: Year, Month

# 🧹 Data Preprocessing

Categorical columns (Product Category, Customer Region, Store Location, etc.) encoded using LabelEncoder

Null values checked

Unnecessary columns dropped before modeling

# 📈 Exploratory Data Analysis (EDA)

Distribution plots for numeric features using Seaborn

Correlation matrix heatmap for understanding feature relationships

# 🔁 Models Trained

## 🟦 Regression (Target: Profit_Margin)

Linear Regression

Random Forest Regression

Evaluation Metrics:

R² Score

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

## 🟥 Classification (Target: High_Profit where Profit_Margin ≥ 20%)

Logistic Regression

Random Forest Classifier

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Feature Importance is visualized for Random Forest models.

# 📦 Dependencies

Install required libraries using:

bash
Copy
Edit
pip install pandas scikit-learn matplotlib seaborn openpyxl

# 🚀 How to Run

Place SalesDB.xlsx in the project directory.

Run the script:

bash
Copy
Edit
python sales_analysis.py

# 📌 Results Summary

Regression models estimate Profit_Margin using sales and cost data.

Classification models predict whether a transaction yields High_Profit.

Random Forest models provide insight into key features influencing profits.

# 🧠 Learnings

Feature engineering significantly impacts model accuracy.

Random Forest models outperform linear/logistic models in both regression and classification.

Profitability is strongly linked to discount, cost, and shipping variables.
