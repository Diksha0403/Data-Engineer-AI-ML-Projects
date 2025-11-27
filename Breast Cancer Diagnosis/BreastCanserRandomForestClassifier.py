###################################################
# Required Python Packages
###################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

###################################################
# File Paths 
###################################################
BORDER="-"*65
File_Name = "breast-cancer-wisconsin.csv" 
Model_Path = "breastcancer_randomforest_pipeline.joblib" 

###################################################
#  Headers 
###################################################

Headers = [ 
"CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", 
"MarginalAdhesion", 
"SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", 
"CancerType" 
] 

###################################################
# Function name :   read_data 
# Description :     Read the data into pandas dataframe  
# Inpt :            path of CSV file   
# Output :          Gives the data
###################################################

def read_data(path):
    """Read the data into pandas dataframe"""
    data = pd.read_csv(path,header=None)
    return data

######################################################## 
# Function name :  get_headers 
# Description :    dataset headers   
# Input :          dataset   
# Output :         Returns the header                
######################################################## 

def get_headers(dataset):
    """Return dataset headers""" 
    return dataset.columns.values

######################################################## 
# Function name :    add_headers 
# Add the headers to the dataset 
# dataset 
# Updated dataset 
########################################################

def add_headers(dataset,headers):
    """Add headers to dataset"""
    dataset.columns=headers
    return dataset

######################################################## 
# Function name :    data_file_to_csv 
# Input :            Nothing 
# Output :           Write the data to CSV 
########################################################

def Load_Dataset():
    dataset = pd.read_csv(File_Name)
    print("File saved ...!")
    return dataset

######################################################## 
# Function name :    handel_missing_values 
#  Description :     Filter missing values from the dataset 
# Input :            Dataset with mising values 
# Output :           Dataset by remocing missing values 
######################################################## 

def handle_missing_values(df):
    """ Show column info and check for null values
    ------------------------------------------------""" 
    print("Data set columns details...")
    print(BORDER)
    print(df.columns)

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Convert the BareNuclei column (and others) to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
   
    # df.dropna(inplace=True)

    print(BORDER)
    """Display basic statistics using .describe()
    ------------------------------------------------"""
    print("Data Frame details statistics")
    print(BORDER)
    print(df.describe())
    print(BORDER)
    return df

#####################################################################################################
#   Function Name   :  findFeaturesAndTarget
#   Description     :  Finds Target and Features
#   Input Params    :  Data Frame
#   Output          :  Features and Targets  
#####################################################################################################
def findFeaturesAndTarget(df):
    features=df.drop(columns=[target])
    target=df[target]
    return features,target 

######################################################## 
# Function name :     split_dataset 
# Description :       Split the dataset with train_percentage 
# Input :             Dataset with related information 
# Output :            Dataset after splitting 
######################################################## 

def split_dataset(dataset, train_percentage, features, target, random_state=42):
    x_train,x_test,y_tarin,y_test = train_test_split(dataset[features], dataset[target],
                                                     train_size=train_percentage,random_state=random_state,stratify=dataset[target])
    return x_train,x_test,y_tarin,y_test


######################################################## 
# Function name :    build_pipeline 
#  Description :      Build a Pipeline: 
#  SimpleImputer:    replace missing with median 
#                    RandomForestClassifier: robust baseline 
########################################################  

def build_pipeline():
    pipe = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("rf",RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight=None
        ))
    ])
    return pipe

######################################################## 
# Function name :  train_pipeline 
#  Description :   Train a Pipeline: 

########################################################  

def train_pipline(pipeline,x_train,y_train):
    pipeline.fit(x_train,y_train)
    return pipeline

########################################################  
# Function name :    save_model 
#  Description :      Save the model 
########################################################  

def save_model(model, path=Model_Path):
    joblib.dump(model,path)
    print(f"Model saved to {path}")

########################################################  
# Function name :    load_model
#  Description :     Load the trained model 
########################################################  

def load_model(path=Model_Path):
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model

########################################################  
# Function name :   plot_confusion_matrix_matshow 
# Description :     Display Confusion Matrix 
######################################################## 

def plot_confusion_matrix_matshow(y_true,y_pred,title="Confusion Matrix"):
    cm = confusion_matrix(y_true,y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    for(i,j),v in np.ndenumerate(cm):
        ax.text(j,i,str(v),ha='center',va='center')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

########################################################  
# Function name :   plot_feature_importances 
# Description :     Display the feture importance 
########################################################

def plot_feature_importances(model,feature_names,title="Feature Importances(Random Forest)"):
    if hasattr(model,"named_steps") and "rf" in model.named_steps:
        rf = model.named_steps["rf"]
        importances = rf.feature_importances_
    elif hasattr(model,"feature_imoprtances_"):
        importances = model.feature_importances_
    else:
        print("Feature importances not available for this model.")
        return
    
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,4))
    plt.bar(range(len(importances)),importances[idx])
    plt.xticks(range(len(importances)),[feature_names[i] for i in idx], rotation=45, ha='right')
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

########################################################  
# Function name :    main 
# Description :      Main function from where execution starts 
######################################################## 

def main():
    """Load and explore data set"""
    dataset=Load_Dataset()

    """ Prepare features/target"""
    features = Headers[1:-1]
    target = "CancerType"

    """Preprocess Data set"""
    dataset = handle_missing_values(dataset)

    """Train and Test Split"""
    x_train,x_test,y_train,y_test = split_dataset(dataset,0.7,features,target)

    print("X_Train Shape:",x_train.shape)
    print("Y_Train Shape:",y_train.shape)
    print("X_Test Shape:",x_test.shape)
    print("Y_Test Shape:",y_test.shape)

    """ Build + Train Pipeline """
    Pipeline = build_pipeline()
    trained_model = train_pipline(Pipeline,x_train,y_train)
    print("Trained Pipeline : ", trained_model)

    """ Predictions """
    predictions = trained_model.predict(x_test)

    """ Metrics """
    print("Train Accuracy : ",accuracy_score(y_train,trained_model.predict(x_train)))
    print("Test Accuracy :",accuracy_score(y_test,predictions))
    print("Classification Report: \n",classification_report(y_test,predictions))
    print("Confusion Matrix :\n",confusion_matrix(y_test,predictions))

    """  Feature importances """
    plot_feature_importances(trained_model,features,title="Feature Importance (RF)")

    """ Save model (Pipeline) using joblib  """
    save_model(trained_model,Model_Path)

    """  Load model and test a sample """
    loaded = load_model(Model_Path)
    sample = x_test.iloc[[0]]
    pred_loaded = loaded.predict(sample)
    print(f"Loaded model prediction for first test sample: {pred_loaded[0]}")

########################################################  
# Application starter 
######################################################## 

if __name__=="__main__":
    main()