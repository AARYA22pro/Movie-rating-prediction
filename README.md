MOVIE RATING PREDICTION WITH PYTHON


1. Build a model that predicts the rating of a movie based on
   features like genre, director, and actors. You can use regression
   techniques to tackle this problem.
2. The goal is to analyze historical movie data and develop a model
   that accurately estimates the rating given to a movie by users or
   critics.
3. Movie Rating Prediction project enables you to explore data
   analysis, preprocessing, feature engineering, and machine
   learning modeling techniques. It provides insights into the factors
   that influence movie ratings and allows you to build a model that
   can estimate the ratings of movies accurately

📌 Features

1.Regression using Random Forest Regressor

2.Classification using:

        *Logistic Regression

        *Random Forest Classifier

        *Support Vector Machine (SVM)

3.Data preprocessing, encoding, and visualization

4.Model evaluation using:

        *Accuracy

        *F1 Score

        *Confusion Matrix

5.Feature importance via:

        *Random Forest

        *SHAP (SHapley Additive exPlanations)

🧠 Objective

To analyze historical IMDb movie data and build predictive models that:

        *Estimate movie ratings (numerical regression)

        *Classify movies into Low / Medium / High rating categories

📂 Dataset

File: IMDb Movies India.csv

Columns Used:

 1.Genre

 2.Director

 3.Actors

 4.Runtime

 5.Rating

Dataset should be placed in the /content folder or adjust file_path accordingly.

📊 Algorithms Used

Task	        Algorithms
Regression	Random Forest Regressor
Classification	Logistic Regression, Random Forest, SVM

🧪 Evaluation Metrics

        *R² Score

        *RMSE

        *Accuracy

        *F1 Score

        *Confusion Matrix


📈 Visualizations

        1.Actual vs Predicted Ratings

        2.Confusion Matrices

        3.Accuracy & F1 Score Comparison

🚀 How to Run

        *pip install pandas numpy matplotlib seaborn scikit-learn shap
Place your IMDb Movies India.csv file in the specified path and run the Python script.

📎 Sample Output

        1.Regression:

                *R² Score: ~0.65

                *RMSE: ~0.80

        2.Classification:

                *Accuracy and F1 Score comparison

                *SHAP plot highlighting top predictive features

🧠 Insights
 
        *Genre and Director have high influence on movie ratings.

        *Random Forest performs consistently well across both tasks.

        *SHAP reveals individual feature contributions to model decisions.

📌 Future Improvements

        *Incorporate more features like Votes, Year, Language

        *Hyperparameter tuning for models

        *Use deep learning models for improved prediction

        *Deploy as a web app using Streamlit or Flask

