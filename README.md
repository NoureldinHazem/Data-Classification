# Data-Classification
Given the MAGIC gamma telescope dataset. This dataset is generated to simulate registration of high energy gamma particles in a ground-based atmospheric
Cherenkov gamma telescope using the imaging technique. The dataset consists of two classes; gammas (signal) and hadrons (background). There are 12332 gamma events and 6688
hadron events. You are required to use this dataset to apply different classification models such as Decision Trees, Naive Bayes Classifier, Random Forests, AdaBoost and KNearest
Neighbor (K-NN). You are also required to tune the parameters of these models, and compare the performance of models with each other.
</br></br>
Data Balancing:</br>
Note that the dataset is class-imbalanced. To balance the dataset, randomly put aside
the extra readings for the gamma “g” class to make both classes equal in size.</br></br>
Data Split:</br>
Split your dataset randomly so that the training set would form 70% of the dataset and
the testing set would form 30% of it.</br></br>
Classification:</br>
Apply the classifiers from the following models on your dataset, tune parameter(s) (if any), compare the performance of models with each other:
1) Decision Tree  [Parameters to be tuned: None]
2) AdaBoost [Parameters to be tuned: n estimators]
3) K-Nearest Neighbors (K-NN) [Parameters to be tuned: K]
4) Random Forests [Parameters to be tuned: n estimators]
5) Naive Bayes [Parameters to be tuned: None]

</br>
Bonus:</br>
Use Pytorch to build a neural network with dense layers and apply the model on your
dataset. Use 2 layers and tune the number of hidden units in every layer.</br></br>
This project was required in AI course</br></br>
This Project was done by:</br>
1. Amr Mohamed Salah</br>
2. Nour El-Din Hazem
