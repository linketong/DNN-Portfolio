# Portfolio optimization with DNN(Deep Neural Network)

A  long-standing interest in finance pertains to the optimal allocation of funds among various financial assets available. 
- Mean-variance analysis, introduced by Markowitz, has long been a popular approach to determining an optimal portfolio's structure and composition. This type of analysis, unfortunately, suffers from several shortcomings. It suggests, in particular, optimal portfolios are independent of an investor's horizon. 
- As initially carried out by Merton, a rigorous dynamic analysis of the consumption-portfolio choice problem reveals some of the missing ingredients. It shows that optimal portfolios should include, in addition to mean-variance terms, dynamic hedging components designed to insure against fluctuations in the opportunity set. 
- An alternative characterization of optimal portfolios is obtained using the martingale approach. The martingale approach helps pinpoint the motivation behind dynamic hedging, namely the stochastic fluctuations in the interest rate and the market price of risk. Random changes in these variables give rise to an interest rate hedge and a market price of risk hedge.

In permissible settings, the martingale approach can be efficiently implemented by Monte Carlo simulation. 
- However, Monte Carlo simulation is not feasible in more sophisticated settings, for example, with recursive utility function is recursive or constraints on portfolios. 
- The main difficulty is that the characterization of the solution often involves FBSDE (forward- backward stochastic differential equations), which are notoriously tricky to handle, both mathematically and computationally. For

This project developed a Deep Neural Network framework to solve the FBSDE resulted from the martingale approach. The motivation is that the success of DNN in many fields has proved its efficiency in handling high-dimensional inputs.As a result We successfully deveolped a DNN scheme to solve a portfolio optimization problem with both recursive utility and constraints. The method is effient in handling dozens of finacial assets.

# Predict Delivery time
The goal of our model is to predict the delivery time for a food delivery company. 

The final prediction for the test data is saved in the file "submission.csv". The two newly appended columns "predicted_duration" and "predicted_delivery_time" are the predicted results.

-   "predicted_duration" has the same meaning with the variable "duration" described above which represents how many seconds it takes to deliver the order since it's created.

-   "predicted_delivery_time" is the date and time when the order is predicted to be delivered.
