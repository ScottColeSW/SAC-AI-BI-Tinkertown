import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

def evaluate_regression(y_actual, y_predicted):
    mse = mean_squared_error(y_actual, y_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_predicted)
    r2 = r2_score(y_actual, y_predicted)
    explained_variance = explained_variance_score(y_actual, y_predicted)
    
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Error:", mae)
    print("Coefficient of Determination (R²):", r2)
    print("Explained Variance Score:", explained_variance)
    
    plt.scatter(y_actual, y_predicted)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Scatter Plot of Actual vs. Predicted Values")
    plt.show()

def evaluate_model(model, features, target):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    predictions = model.predict(features)
    
    accuracy = accuracy_score(target, predictions)
    precision = precision_score(target, predictions)
    recall = recall_score(target, predictions)
    f1 = f1_score(target, predictions)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    # successful_rows = identify_successes(data, threshold)
    # if not successful_rows.empty:
    #     print("Operating within constraints: ")
    #     print(successful_rows)

    # problematic_rows = identify_problems(data, threshold)
    # if not problematic_rows.empty:
    #     print("Potential problems in the manufacturing process:")
    #     print(problematic_rows)
    
    # ai_problematic_rows = ai_identify_problems(data, threshold)
    # if not ai_problematic_rows.empty:
    #     print("AI Potential Warnings:")
    #     print(ai_problematic_rows)