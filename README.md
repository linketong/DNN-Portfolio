# Portfolio optimization with DNN(Deep Neural Network)

This project developed a Deep Neural Network framework to solve the dynamic portfolio optimization problem with Recursive Utility and high-dimensional setting( multiple assets traded in financial market governed by a multivariate state variable). The motivation is that the success of DNN in many fields has proved its efficiency in handling high-dimensional inputs.
- Demo of dBSDE: a jupyter notebook shows some demos of the DNN method
- config.py: set up the parameters in the recursive utility function
- equation.py: the characterization of the FBSDEs that needed to be solve
- solver.py: the construction of the DNN that is used to solve the FBSDE.

# Predict Delivery time
The goal of our model is to predict the delivery time for a food delivery company. 

The final prediction for the test data is saved in the file "submission.csv". The two newly appended columns "predicted_duration" and "predicted_delivery_time" are the predicted results.

-   "predicted_duration" has the same meaning with the variable "duration" described above which represents how many seconds it takes to deliver the order since it's created.

-   "predicted_delivery_time" is the date and time when the order is predicted to be delivered.
