# CREDIT CARD ACCEPATANCE
# Importing application csv
import pandas as pd
df = pd.read_csv('C:\\Users\\jesse\\Downloads\\Project\\Dataset\\application_record.csv')
df

#merge df and df1 
import pandas as pd
merged_df = pd.merge(df , df1 ,on = 'ID',how = 'inner')
merged_df

#checking the duplicate records in credit csv in 'ID' category
duplicate = merged_df.duplicated('ID')
duplicate_count = duplicate.value_counts()
duplicate_count

#Removing the duplicate ID 
merged_df.drop_duplicates(subset = ['ID'])

#Data transformation for EMPLOYEE DAYS
import numpy as np
merged_df['EMPLOYEE STATUS'] = np.where(merged_df['EMPLOYEE DAYS'] < 0, 'EMPLOYED', 'NON EMPLOYED')
merged_df

#Removing the duplicate ID 
filtered_df = merged_df.drop_duplicates(subset = ['ID'])
filtered_df

#Checking the Sub category inside a Variable
filtered_df['EMPLOYEE STATUS'].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate IQR for 'INCOME TOTAL'
Q1 = filtered_df['INCOME TOTAL'].quantile(0.25)
Q3 = filtered_df['INCOME TOTAL'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = (filtered_df['INCOME TOTAL'] < lower_bound) | (filtered_df['INCOME TOTAL'] > upper_bound)

# Count the number of outliers
outliers_count = outliers.sum()

# Display count of outliers
print("Number of outliers in 'INCOME TOTAL':", outliers_count)

# Display a box plot for 'INCOME TOTAL' in the DataFrame
plt.figure(figsize=(10, 8), dpi=150)
sns.boxplot(x=filtered_df['INCOME TOTAL'])

# Highlight outliers on the box plot
plt.scatter([], [], label=f'Outliers: {outliers_count}', color='red', marker='o')
plt.legend()

plt.title('Box Plot for INCOME TOTAL with Outliers')
plt.show()


#Removed all the outliers

Q1 = filtered_df['INCOME TOTAL'].quantile(0.25)
Q3 = filtered_df['INCOME TOTAL'].quantile(0.75)
IQR = Q3 - Q1

outliers = (filtered_df['INCOME TOTAL'] < (Q1 - 1.5 * IQR)) | (filtered_df['INCOME TOTAL'] > (Q3 + 1.5 * IQR))

cleaned_df = filtered_df[~outliers]
cleaned_df


#Removing the null values
import pandas as pd

# Assuming 'cleaned_df' is your DataFrame
# You can replace 'cleaned_df' with your actual DataFrame name

# Remove rows where 'OCCUPATION TYPE' is null
cleaned_df = cleaned_df.dropna(subset=['OCCUPATION TYPE'])

# Display the updated DataFrame
print(cleaned_df)


import numpy as np

# Assuming 'BIRTH' column contains the birthdate in days
cleaned_df['AGE'] = np.floor(abs(cleaned_df['BIRTH'] / 365.25))


import pandas as pd

# Assuming 'cleaned_df' is your DataFrame and you want to remove columns 'column1' and 'column2'
columns_to_remove = ['EMPLOYEE DAYS', 'MONTHS_BALANCE', 'BIRTH', 'EMPLOYEE STATUS', 'EMPLOYEE DAYS']

cleaned_df.drop(columns=columns_to_remove, inplace=True)

#Checking the Sub category inside a Variable
A = cleaned_df['OCCUPATION TYPE'].value_counts()
B = cleaned_df['EDUCATION TYPE'].value_counts()
C = cleaned_df['INCOME TYPE'].value_counts()
D = cleaned_df['HOUSING TYPE'].value_counts()
E = cleaned_df['FAMILY STATUS'].value_counts()
F = cleaned_df['STATUS'].value_counts()

import pandas as pd
from scipy.stats import chi2_contingency

# Assuming 'final_df' is your DataFrame with categorical variables
contingency_table = pd.crosstab(cleaned_df['FAMILY STATUS'], cleaned_df['HOUSING TYPE'])

# Perform Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Value: {chi2}")
print(f"P-value: {p}")


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
# If your DataFrame contains both numerical and categorical data, consider encoding categorical variables first

# Create a correlation matrix
correlation_matrix = cleaned_df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap using seaborn
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Show the plot
plt.title("Correlation Matrix")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
# Replace 'variable1' and 'variable2' with the names of your continuous variables

# Set up the matplotlib figure
plt.figure(figsize=(10, 10))

# Create a scatter plot
sns.scatterplot(x='AGE', y='INCOME TOTAL', data=cleaned_df)

# Add labels and title
plt.xlabel('AGE')
plt.ylabel('INCOME TOTAL')
plt.title('Scatter Plot of Continuous Variables')

# Show the plot
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame and 'variable' is the column for which you want to create a KDE plot

# Set up the matplotlib figure
plt.figure(figsize=(10, 6))

# Create a Kernel Density Plot
sns.kdeplot(cleaned_df['INCOME TOTAL'], shade=True)

# Add labels and title
plt.xlabel('Variable')
plt.ylabel('Density')
plt.title('Kernel Density Plot')

# Show the plot
plt.show()

import pandas as pd

# Assuming 'cleaned_df' is your existing DataFrame
# Assuming 'STATUS' is the column for which you want to create a binary variable

# Create a binary variable for 'STATUS'
cleaned_df['STATUS'] = cleaned_df['STATUS'].replace(['X', 'C', '0'], 0).replace(['1', '2', '3', '4', '5'], 1).astype(int)

# Identify categorical columns in the DataFrame
categorical_columns = ['GENDER', 'CAR', 'PROPERTY', 'INCOME TYPE', 'EDUCATION TYPE', 'FAMILY STATUS', 'HOUSING TYPE', 'OCCUPATION TYPE']

# Create dummy variables for categorical columns
dummy_variables = pd.get_dummies(cleaned_df[categorical_columns], prefix=categorical_columns)

# Concatenate the dummy variables with the original DataFrame
cleaned_df = pd.concat([cleaned_df, dummy_variables], axis=1)

# Drop the original categorical columns if needed
cleaned_df = cleaned_df.drop(categorical_columns, axis=1)

# Display the updated DataFrame
print(cleaned_df)

from sklearn.tree import DecisionTreeClassifier

# Assuming 'X_train_resampled' and 'y_train_resampled' are the resampled training data
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test data
y_pred_dt = model_dt.predict(X_test)

# Evaluate the decision tree model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
classification_report_dt = classification_report(y_test, y_pred_dt)

# Display results for the decision tree model
print(f'Decision Tree Accuracy: {accuracy_dt:.4f}')
print('Decision Tree Classification Report:\n', classification_report_dt)

from sklearn.ensemble import RandomForestClassifier

# Assuming 'X_train_resampled' and 'y_train_resampled' are the resampled training data
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test data
y_pred_rf = model_rf.predict(X_test)

# Evaluate the random forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_report_rf = classification_report(y_test, y_pred_rf)

# Display results for the random forest model
print(f'Random Forest Accuracy: {accuracy_rf:.4f}')
print('Random Forest Classification Report:\n', classification_report_rf)


# Assuming 'model_rf' is your trained random forest model
feature_importance = model_rf.feature_importances_

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({'Feature': X_train_resampled.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top features
print(feature_importance_df.head(10))


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Assuming 'cleaned_df' is your DataFrame
# Assuming 'STATUS' is the dependent variable, and other columns are independent variables

# Specify independent variables (including dummy variables and numerical variable 'INCOME')
independent_variables = ['GENDER_M', 'GENDER_F', 'CAR_N', 'CAR_Y', 
                         'INCOME TYPE_Working', 'INCOME TYPE_Pensioner', 'INCOME TYPE_State servant',
                         'INCOME TYPE_Commercial associate', 'INCOME TYPE_Student', 
                         'EDUCATION TYPE_Higher education', 'EDUCATION TYPE_Secondary / secondary special',
                         'FAMILY STATUS_Married', 'FAMILY STATUS_Single / not married', 'FAMILY STATUS_Widow',
                         'HOUSING TYPE_House / apartment', 'HOUSING TYPE_Rented apartment', 'HOUSING TYPE_With parents',
                         'OCCUPATION TYPE_Laborers', 'OCCUPATION TYPE_Core staff', 'OCCUPATION TYPE_Drivers',
                         'OCCUPATION TYPE_Managers', 'OCCUPATION TYPE_High skill tech staff',
                         'AGE', 'FAMILY SIZE', 'NO OF CHILDREN', 'PHONE', 'EMAIL', 'INCOME TOTAL']

# Specify the dependent variable
dependent_variable = 'STATUS'

# Create the feature matrix (X) and target variable (y)
X = cleaned_df[independent_variables]
y = cleaned_df[dependent_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression before SMOTE
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_prob_lr = model_lr.predict_proba(X_test)[:, 1]

# Random Forest after SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train_resampled, y_train_resampled)
y_prob_rf = model_rf.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area for Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Compute ROC curve and ROC area for Random Forest after SMOTE
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC curves
plt.figure(figsize=(10, 6))

plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label='Logistic Regression (area = {:.2f})'.format(roc_auc_lr))
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='Random Forest with SMOTE (area = {:.2f})'.format(roc_auc_rf))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

import matplotlib.pyplot as plt

# Assuming 'model_rf' is your trained random forest model
feature_importance = model_rf.feature_importances_

# Check the lengths of both arrays
print('Length of cleaned_df.columns:', len(cleaned_df.columns))
print('Length of feature_importance:', len(feature_importance))

# Ensure the lengths match before creating the DataFrame
if len(cleaned_df.columns) == len(feature_importance):
    # Create a DataFrame to display feature importance
    feature_importance_df = pd.DataFrame({'Feature': cleaned_df.columns, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.show()
else:
    print("Error: The lengths of 'cleaned_df.columns' and 'feature_importance' do not match.")








