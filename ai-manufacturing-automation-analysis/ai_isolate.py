
# Problem identification function
def identify_problems(dataframe, threshold):
    model = IsolationForest(contamination=threshold)  # Adjust contamination based on your data
    model.fit(dataframe)

    anomalies = model.predict(dataframe)
    problematic_indices = [i for i, anomaly in enumerate(anomalies) if anomaly == -1]

    problematic_rows = dataframe.iloc[problematic_indices]

    return problematic_rows

# Problem identification function
def identify_successes(dataframe, threshold):
    model = IsolationForest(contamination=threshold)  # Adjust contamination based on your data
    model.fit(dataframe)

    anomalies = model.predict(dataframe)
    successful_indices = [i for i, anomaly in enumerate(anomalies) if anomaly == 1]  # Select instances classified as normal

    successful_rows = dataframe.iloc[successful_indices]

    return successful_rows

# Problem identification function
def ai_identify_problems(dataframe, threshold):
    # features = ["part_temperature", 
    # "machine_speed", 
    # "measurement_accuracy", 
    # "passenger_count", 
    # "time_savings", 
    # "emergency_signals", 
    # "fuel_efficiency", 
    # "emission_reduction", 
    # "operational_costs"
    # ]

    model = IsolationForest(contamination=threshold)  # You can adjust hyperparameters as needed
    model.fit(dataframe[features], dataframe["flight_problems","Mechanical_Warning_Feature", "Cost_Inefficiencies_Warning_Feature","Emergency_Signals_Warning_Feature"])  # Assuming "flight_problems" is your target column

    predictions = model.predict(dataframe[features])
    problematic_indices = [i for i, prediction in enumerate(predictions) if prediction == 1]

    problematic_rows = dataframe.iloc[problematic_indices]

    return problematic_rows

