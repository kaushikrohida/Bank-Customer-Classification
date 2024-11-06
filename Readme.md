# Bank Customer Churn Classification (Prediction) Project

## Overview

This project aims to predict whether a customer will leave the bank (churn) based on various features like age, credit score, balance, and more. This classification task uses machine learning models to help identify customers who are at risk of churning, allowing the bank to implement targeted strategies to retain them.

The dataset used in this project contains information on 10,000 customers, including demographics, account balance, activity, and product usage details.

## Table of Contents

- Dataset
- Project Structure
- Usage
- Models and Evaluation
- Results
- Conclusion
- Contributing

## Dataset

The dataset (Bank_Churn.csv) contains the following 13 columns:

- **CustomerId**: Unique identifier for the customer.
- **Surname**: Customer's surname.
- **CreditScore**: Credit score of the customer.
- **Geography**: Country of the customer (e.g., France, Spain).
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: Account balance of the customer.
- **NumOfProducts**: Number of products held by the customer.
- **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No).
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No).
- **EstimatedSalary**: Estimated annual salary of the customer.
- **Exited**: Whether the customer has left the bank (1 = Yes, 0 = No).

The target variable for this classification task is **Exited**, indicating whether the customer churned.

## Project Structure

The project is structured as follows:

- **Bank Churn Classification.ipynb**: The Jupyter Notebook containing the data exploration, preprocessing, model building, and evaluation steps.
- **Bank_Churn.csv**: The dataset used for training and evaluating the model.
- **README.md**: This file, providing an overview and instructions for the project.

## Usage

To run the project:

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Bank\ Churn\ Classification.ipynb
   ```

2. Follow along with the steps in the notebook to explore the dataset, preprocess the data, build the models, and evaluate their performance.

The notebook covers the following steps:

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Building (using algorithms like Logistic Regression, Decision Tree, Random Forest, etc.)
- Evaluation Metrics (accuracy, precision, recall, F1-score)

## Models and Evaluation

Several machine learning models were trained to predict customer churn, including:

- **Logistic Regression**: This model was used as a baseline to understand the relationship between features and the target variable. Logistic Regression is easy to interpret and provides insights into the importance of different features through coefficients. However, it performed moderately well, with an accuracy of around 78%. It was observed that the model had a relatively lower recall, which means it missed some customers who actually churned.

- **Decision Tree Classifier**: The Decision Tree model provided better interpretability and was able to capture non-linear relationships between features. The model had an accuracy of around 80%, but it was prone to overfitting due to its tendency to create complex trees that perfectly classify the training data. To address this, hyperparameter tuning (such as setting the maximum depth) was applied to improve generalization.

- **Random Forest Classifier**: The Random Forest model, which is an ensemble of multiple decision trees, performed significantly better. It achieved an accuracy of 86%, and it balanced both precision and recall. By aggregating the results of multiple trees, Random Forest reduced the variance and improved robustness. Feature importance scores provided by Random Forest showed that Age, Credit Score, and Balance were key factors in determining churn.

- **Gradient Boosting Classifier**: Gradient Boosting was also employed to enhance performance further. This model performed similarly to Random Forest, with an accuracy of 85%. However, Gradient Boosting tends to be slower to train compared to Random Forest, especially for large datasets, due to its sequential nature of building trees. The model showed slightly better precision, meaning it was more conservative in predicting churn, thus reducing false positives.

## Hyperparameter Tuning

For each model, hyperparameter tuning was performed using GridSearchCV to identify the optimal parameters. For instance:

- **Random Forest**: Parameters such as the number of estimators, maximum depth, and minimum samples split were tuned.
- **Gradient Boosting**: Learning rate, number of estimators, and maximum depth were adjusted to achieve better performance.

## Cross-Validation

To ensure that the models were not overfitting and to evaluate their generalization capabilities, cross-validation was employed. A 5-fold cross-validation strategy was used, where the dataset was split into five parts, and the model was trained and validated five times, each time using a different part as the validation set.

## Evaluation Metrics

The models were evaluated using the following metrics:

- **Accuracy**: The ratio of correctly predicted observations to the total observations.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives. Precision is useful to measure how many predicted churns were actually correct.
- **Recall**: The ratio of correctly predicted positive observations to all the observations in the actual class. This metric is crucial for identifying all customers at risk of churning.
- **F1-Score**: The harmonic mean of precision and recall. It provides a balance between precision and recall, making it useful when there is an uneven class distribution.
- **ROC-AUC Score**: The Receiver Operating Characteristic - Area Under the Curve score was used to evaluate the models' performance across different thresholds, giving a better understanding of the true positive rate versus the false positive rate.

The Random Forest Classifier was ultimately selected as the best model due to its high accuracy, balanced precision and recall, and the ability to provide feature importance insights. The final model was also evaluated on a held-out test set to ensure its performance was consistent.

## Results

The key findings of this project include:

- The most important factors affecting customer churn were Credit Score, Age, Geography, and Balance.
- The Random Forest Classifier performed the best, achieving an accuracy of 86%, with balanced precision and recall scores.
- The project demonstrates how machine learning can help banks predict customer churn and potentially reduce churn by targeting at-risk customers with appropriate interventions.

## Conclusion

In conclusion, this project successfully demonstrated the use of various machine learning models to predict customer churn in the banking sector. The Random Forest Classifier emerged as the best-performing model, providing valuable insights into the factors influencing customer churn, such as Credit Score, Age, Geography, and Balance. By using these insights, banks can take proactive steps to retain at-risk customers through targeted marketing campaigns or personalized customer service.

The project also highlighted the importance of feature engineering, hyperparameter tuning, and model evaluation in building a robust predictive model. Future work could include incorporating additional data sources, such as customer transaction history, or testing advanced deep learning models to further improve prediction accuracy.

Overall, this project provides a solid foundation for understanding customer churn and demonstrates the potential of machine learning to create impactful business solutions.

## Contributing

Contributions are welcome! If you have ideas for improving the project, feel free to fork the repository and submit a pull request.

## Contact

For questions or suggestions, please contact [Kaushik Rohida](mailto:rohidakaushik@gmail.com).