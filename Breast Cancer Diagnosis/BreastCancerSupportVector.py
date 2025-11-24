from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

border = "*"*50

def MarvellousSVM():
    # Load Dataset
    cancer = datasets.load_breast_cancer()

    #print the name of the 13 features
    print("Features of the cancer dataset :",cancer.feature_names)
    print(border)

    #print the label type of cancer('malignant''benign')
    print("Label of the cancer dataset: ",cancer.target_names)
    print(border)

    #print data(featre)shape
    print("Shape of dataset is: ",cancer.data.shape)
    print(border)

    #print the cancer data features (top 5 records)
    print("First 5 records are : ")
    print(cancer.data[0:5])
    print(border)

    #print the cancer labels (0:malignant, 1:benign)
    print("Target of dataset: ",cancer.target)
    print(border)

    #Split dataset into training set and test set
    x_train,x_test,y_tarin,y_test=train_test_split(cancer.data,cancer.target,test_size=0.3,random_state=109)

    #create a svm classifier
    clf = svm.SVC(kernel='linear') 

    #Train the model using training sets
    clf.fit(x_train,y_tarin)

    #Predict the response for test dataset
    y_pred = clf.predict(x_test)

    #Model Accuracy: How often is the classifier correct ?
    print(border)
    print("Accuracy of the model is: ",metrics.accuracy_score(y_test,y_pred)*100)


def main():
    print("____________Marvellous Support Vector Machine____________")

    MarvellousSVM()


if __name__=="__main__":
    main()