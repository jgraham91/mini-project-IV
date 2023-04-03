# Mini-project IV

### [Assignment](assignment.md)

## Project/Goals
Create a model that accurately predicts if an individual will get approved for a loan.

## Hypothesis
Before doing any EDA my hypothesis is that loan approval will be most correlated with income amount and education history.

## EDA 
In my EDA I found out that the highest level of correlation has to do with the credit history.

## Process

After exploring the data I found out where my Nan Values were.  For the Loan Amount values I imputer with the mean.  For all the categorical values that were missing I imputed with the most frequent.

I split the data into numerical and categorical features.    For the numerical features I did a logorithmic transform then a standardization.  For the categorical features I one hot encoded them, then did a PCA to reduce dimenisonality.  I initially did this step by step, then I made a pipeline.

I tested a SVM, random forest classifier and logistic regression model.   The SVM gave the highest accuracy.

I picked the model and made a flask api.  You can see at the bottom of the notebook.

## Results/Demo

 I got the best results from my SVM model.  The accuracy was almost 0.8.    However I found that it was a horrible predicter outside of the range of loan amounts that it was trained on.   The highest loan was $7000 and above that the model always said yes to approval.  In reality someone with no credit history would not get a $2,000,000 loan making for a very poor model, although it had the highest accuracy.  Within the 0 to $7000 range the best indicator of wheather you were aproved for a loan was whether or not you had a credit history.   I also had credit history and credit loan term as numerical features in my model, but i think they make more sense as categorical variables.


## Challanges 

It was chalenging to get the flask api to work with the final pickled model.


## Future Goals
If I had more time I would re run the project with the logistic regressor as my model and see if it was a better predictor or high loans, or if it would have the same inherent flaw as the SVM model.