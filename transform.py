import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Global variables to store values from training data
fare_medians_by_pclass = {}
overall_fare_median = None
age_medians_by_group = {}
overall_age_median_train = None # Added for fallback age imputation
embarked_mode = None
scaler = StandardScaler()
# Store OHE column names from training data
title_ohe_columns = []
embarked_ohe_columns = []

def transform(source, target, train):
    global fare_medians_by_pclass, overall_fare_median, age_medians_by_group, embarked_mode, scaler
def transform(source, target, train):
    global fare_medians_by_pclass, overall_fare_median, age_medians_by_group, overall_age_median_train, embarked_mode, scaler
    global title_ohe_columns, embarked_ohe_columns
    d = pd.read_csv(source)
    
    # Handle Fare imputation
    if train:
        overall_fare_median = d['Fare'].median()
        fare_medians_by_pclass = d.groupby('Pclass')['Fare'].median().to_dict()
        
        for pclass_val, median_fare in fare_medians_by_pclass.items():
            d.loc[(d['Fare'].isna()) & (d['Pclass'] == pclass_val), 'Fare'] = median_fare
        if d['Fare'].isna().any(): 
            d['Fare'].fillna(overall_fare_median, inplace=True)
    else:
        for pclass_val, median_fare in fare_medians_by_pclass.items():
            d.loc[(d['Fare'].isna()) & (d['Pclass'] == pclass_val), 'Fare'] = median_fare
        if d['Fare'].isna().any(): 
            d['Fare'].fillna(overall_fare_median, inplace=True)

    # Create FamilySize & IsAlone
    d['FamilySize'] = d['SibSp'] + d['Parch'] + 1
    d['IsAlone'] = 0
    d.loc[d['FamilySize'] == 1, 'IsAlone'] = 1

    # Extract Title
    d['Title'] = d['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    d['Title'] = d['Title'].replace(['Mlle', 'Ms', 'Lady'], 'Miss') # Consolidating Miss variations
    d['Title'] = d['Title'].replace(['Mme', 'Countess', 'Dona'], 'Mrs')  # Consolidating Mrs variations
    d['Title'] = d['Title'].replace(['Sir', 'Don', 'Jonkheer'], 'Mr') # Consolidating Mr variations
    # Group rare or less distinct titles into 'Other'
    rare_titles = ['Dr', 'Rev', 'Major', 'Col', 'Capt'] 
    d['Title'] = d['Title'].apply(lambda x: 'Other' if x in rare_titles or x not in ['Mr', 'Miss', 'Mrs', 'Master'] else x)
    
    title_dummies = pd.get_dummies(d['Title'], prefix='Title')
    d = pd.concat([d, title_dummies], axis=1)
    if train:
        title_ohe_columns = title_dummies.columns.tolist()
    else:
        d = d.reindex(columns=d.columns.tolist() + [col for col in title_ohe_columns if col not in d.columns], fill_value=0)


    # Improved Age Imputation
    if train:
        # Calculate overall_age_median_train first for comprehensive fallback
        overall_age_median_train = d['Age'].median() 
        age_medians_by_group = d.groupby(['Pclass', 'Sex', 'Title'])['Age'].median().to_dict()
        
        for group_keys, median_age in age_medians_by_group.items():
            condition = (d['Age'].isna()) & \
                        (d['Pclass'] == group_keys[0]) & \
                        (d['Sex'] == group_keys[1]) & \
                        (d['Title'] == group_keys[2])
            d.loc[condition, 'Age'] = median_age
        # Fallback for any remaining NaNs after group-based imputation
        if d['Age'].isna().any():
            d['Age'].fillna(overall_age_median_train, inplace=True) 
    else:
        for group_keys, median_age in age_medians_by_group.items():
            condition = (d['Age'].isna()) & \
                        (d['Pclass'] == group_keys[0]) & \
                        (d['Sex'] == group_keys[1]) & \
                        (d['Title'] == group_keys[2])
            d.loc[condition, 'Age'] = median_age
        # Fallback for groups not in training data or other NaNs, using overall training median
        if d['Age'].isna().any():
            d['Age'].fillna(overall_age_median_train, inplace=True)


    # Handle Embarked
    if train:
        embarked_mode = d['Embarked'].mode()[0]
        d['Embarked'].fillna(embarked_mode, inplace=True)
    else:
        d['Embarked'].fillna(embarked_mode, inplace=True) # Use mode from training data
    
    embarked_dummies = pd.get_dummies(d['Embarked'], prefix='Embarked')
    d = pd.concat([d, embarked_dummies], axis=1)
    if train:
        embarked_ohe_columns = embarked_dummies.columns.tolist()
    else:
        d = d.reindex(columns=d.columns.tolist() + [col for col in embarked_ohe_columns if col not in d.columns], fill_value=0)

    # Transform Sex to numeric
    d['Sex'] = d['Sex'].map({'male': 0, 'female': 1}).astype(int)
    
    # Define numerical features for scaling
    numerical_features = ['Age', 'Fare', 'FamilySize', 'Pclass', 'SibSp', 'Parch'] 
    
    if train:
        scaler.fit(d[numerical_features])
    # Ensure numerical features exist before scaling (e.g. if a file is empty or malformed)
    # This is more of a safeguard; typically columns should exist from CSV read.
    existing_numerical_features = [col for col in numerical_features if col in d.columns]
    if existing_numerical_features: # only scale if there are numerical features
        d[existing_numerical_features] = scaler.transform(d[existing_numerical_features])

    # Define final columns
    final_columns = ['PassengerId']
    if train:
        final_columns += ['Survived']
        
    # Add scaled numerical features + Sex (already 0/1) + IsAlone (0/1)
    final_columns += existing_numerical_features # Use existing_numerical_features to be safe
    final_columns += ['Sex', 'IsAlone'] 
    
    # Add OHE columns, ensuring consistency
    final_columns += title_ohe_columns
    final_columns += embarked_ohe_columns
    
    # Ensure all selected columns exist in the DataFrame, remove duplicates, maintain order
    # This handles cases where some OHE columns might not be generated if a category is absent
    # The reindex step for OHE columns should handle this, but as a safeguard:
    seen = set()
    final_columns_ordered_unique = [x for x in final_columns if not (x in seen or seen.add(x))]
    
    # Filter to columns that actually exist in d, preserving order
    # This is critical because test set might not have all 'Survived' or some rare OHE if not handled by reindex
    final_df_columns = [col for col in final_columns_ordered_unique if col in d.columns]
    
    d = d[final_df_columns] # Select final columns
    
    d.to_csv(target, header=False, index=False)

if __name__ == '__main__':
    # Ensure the script can find the CSV files.
    # Assuming 'train.csv' and 'test.csv' are in the same directory as the script.
    try:
        transform('train.csv','train_transformed.csv', train=True)
        transform('test.csv','test_transformed.csv', train=False)
        print("Transformation complete. Output files: train_transformed.csv, test_transformed.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure 'train.csv' and 'test.csv' are in the script's directory.")
    except Exception as e:
        print(f"An error occurred during transformation: {e}")
