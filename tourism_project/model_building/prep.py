import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#read the dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

#Replacing "Fe Male" in Gender column with "Female"
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

#Replacing "Unmarried" in MaritalStatus column with "Single"
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

# Feature Engineering Age into AgeGroup

# Define bin edges and corresponding labels.np.inf handles any age above 76
bins = [18, 25, 41, 57, np.inf]
labels = ['18-25', '26-41', '42-57', '58-76+']

# Create the new AgeGroup column
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)

#Dropping index, CustomerID and Age as they don't add any value.
df.drop(columns=["Unnamed: 0","CustomerID","Age"], inplace=True)

# NOTE: 'TypeofContact','Occupation', 'Gender','ProductPitched','MaritalStatus'
# and 'Designation' are intentionally left as raw strings.
# The training pipeline one-hot-encodes it, and the Streamlit app also sends
# raw values. Encoding it here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("TypeofContact values kept as:", sorted(X["TypeofContact"].unique()))
print("Occupation values kept as:", sorted(X["Occupation"].unique()))
print("Gender values kept as:", sorted(X["Gender"].unique()))
print("ProductPitched values kept as:", sorted(X["ProductPitched"].unique()))
print("MaritalStatus values kept as:", sorted(X["MaritalStatus"].unique()))
print("Designation values kept as:", sorted(X["Designation"].unique()))
