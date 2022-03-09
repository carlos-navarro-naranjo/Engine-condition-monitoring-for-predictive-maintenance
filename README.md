# Engine-condition-monitoring-for-predictive-maintenance
The code predicts whether the engines run fine or if they operate in diminished capacity.



Binary Classification 

Dataset 
The dataset has two attributes, EV1 and EV2, and one output, Status.

If Status==1, the engine is running fine and needs no further analysis. 
If Status==0, the engine is operating in diminished capacity and engineers should perform a more detailed analysis to determine remaining engine life. 

 
Logistic Regression Code using holdout validation
•	Instantiates a logistic regression model
•	Trains the model using X_train and y_train
•	Displays the learned weights to the console using the print function
•	Creates a plot to visualize the data and learned decision boundary
•	Evaluates the model on the test set (X_test and y_test), using an appropriate ConfusionMatrixDisplay class function to generate a confusion matrix 
