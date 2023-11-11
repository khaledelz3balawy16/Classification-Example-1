# Classification-Example-1
This project applies the logistic regression algorithm to a problem of classifying students into accepted or rejected based on their scores in two exams.

Data File
The "ex2data1.txt" data file is used, containing two columns for exam grades and one column for classification (accepted or not).

Main Steps
Read Data:

Data is read using the Pandas library.
Visualize Data:

Data is visualized on a graph to observe the distribution between accepted and rejected students.
Define Sigmoid Function:

The sigmoid function is defined and used to transform values into a range between 0 and 1.
Define Cost Function:

The cost function is defined to measure the accuracy of predictions.
Prepare Data:

A column of ones is added to facilitate mathematical operations.
Define X and y:

The data matrix and target results are defined.
Calculate Model Cost Before Optimization:

The cost of the model is calculated before the optimization process begins.
Define Gradient Descent Function:

The partial gradient function is defined for use in optimization.
Optimization:

Model parameters are improved using the truncated Newton conjugate-gradient technique.
Calculate Model Accuracy:

Model accuracy is calculated based on predictions and actual results.
Credits
The Pandas library was used for data reading.
The Matplotlib library was used for graph plotting.
The Scipy library was used to execute the optimization process.
