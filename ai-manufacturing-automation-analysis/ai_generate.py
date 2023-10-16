import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_histograms(data):
    data.hist(figsize=(15, 15))
    plt.show()

def plot_correlation_matrix(data):
    corr_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

def plot_feature_importance(model, features):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.feature_importances_, y=features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance Plot')
    plt.show()

def plot_scatter(feature, target, data):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=feature, y=target, data=data)
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f'Scatter Plot of {feature} vs. {target}')
    plt.show()


def generate_synthetic_data(num_points):
    # setup the default tolerances
    #mechanical performance issues
    #Mechanical_Warning_Feature
    parts_temp_tolerance_max = 1000
    machine_speed_tolerance = 3400
    fuel_efficiency_min = 50
    battery_health_min = .2

    #cost performance issues
    #Cost_Inefficiencies_Warning_Feature
    operational_cost_max = 240
    time_savings_min = 100

    #too many warnings
    #Emergency_Signals_Warning_Feature
    emergency_signals_alert = 350
    #uses batter_health_min also

    # setup the data to be used
    data = {
        "passenger_count": [],
        "time_savings": [],
        "emergency_signals": [],
        "fuel_efficiency": [],
        "emission_reduction": [],
        "operational_costs": [],
        "part_temperature": [],
        "machine_speed": [],
        "measurement_accuracy": [],
        "battery_health": [],
        "motor_efficiency": [],
        "flight_time": [],
        "flight_range": [],
        "noise_level": [],
        "charging_time": [],
        "maintenance_intervals": [],
        "avionics_performance": [],
        "component_reliability": [],
        "energy_consumption": [],
        "regulatory_compliance": [],
        "passenger_load_factors": [],
        "weather_impact": [],
        "material_usage_efficiency": [],
        "carbon_emissions": [],
        "time_savings_as_compared_driving": [],
        "payload_capacity": [],
        "emergency_landing_success_rate": [],
        "time_to_emergency_landing": [],
        "safety_margin_for_emergency_landing": [],
        "terrain_analysis": [],
        "weather_impact_on_emergency_landing": [],
        "system_redundancy": [],
        "pilot_training_and_response": [],
        "passenger_safety_during_emergency": [],
        "communication_and_ground_support": [],
        "cabin_alertness": [],
        "flight_problems": [],
        "Mechanical_Warning_Feature": [],
        "Cost_Inefficiencies_Warning_Feature": [],
        "Emergency_Signals_Warning_Feature": [],
        "Engine_Performance_Feature": [],
        "Battery_Health_Feature": [],
        "Flight_Efficiency_Feature": [],
        "Emergency_Signals_Feature": [],
        "Weather_Impact_Feature": [],
        "Terrain_Analysis_Feature": [],
        "Safety_Features": [],
        "Mechanical_Warning_Feature": [],
        "Cost_Inefficiencies_Warning_Feature": [],
        "Emergency_Signals_Warning_Feature": [],
        "General_Warning_Feature": [],
        "turbulence_intensity": []
    }

    np.random.seed()
    for _ in range(num_points):
        time_savings = np.random.randint(30, 2000)
        emergency_signals = np.random.randint(1, 500)
        fuel_efficiency = np.random.randint(10, 150)
        emission_reduction = np.random.randint(100, 2000)
        operational_costs = np.random.randint(10, 4000)
        part_temperature = np.random.randint(100, 3200)
        machine_speed = np.random.randint(1000, 14000)
        measurement_accuracy = np.random.randint(1, 10)
        battery_health = np.random.uniform(0.001, 1.0)
        motor_efficiency = np.random.uniform(0.05, 1.0)
        flight_time = np.random.uniform(20, 360)
        flight_range = np.random.uniform(.1, 200)
        noise_level = np.random.uniform(5, 800)
        charging_time = np.random.uniform(1, 45)
        maintenance_intervals = np.random.randint(100, 500)
        avionics_performance = np.random.uniform(0.09, 1.0)
        component_reliability = np.random.uniform(0.95, 1.0)
        energy_consumption = np.random.uniform(500, 8000)
        regulatory_compliance = np.random.uniform(0.8, 1.0)
        passenger_load_factors = np.random.uniform(0.1, 0.9)
        weather_impact = np.random.uniform(0.6, 1.0)
        material_usage_efficiency = np.random.uniform(0.85, 1.0)
        carbon_emissions = np.random.uniform(10, 700)
        operational_costs = np.random.uniform(150, 3000)
        passenger_count = np.random.randint(1, 5)
        time_savings_as_compared_driving = np.random.uniform(20, 12000)
        payload_capacity = np.random.uniform(10, 1000)
        emergency_landing_success_rate = np.random.uniform(0.6, 1.0)
        time_to_emergency_landing = np.random.uniform(1, 5000)
        safety_margin_for_emergency_landing = np.random.uniform(150, 500)
        terrain_analysis = np.random.uniform(0.5, 1.0)
        weather_impact_on_emergency_landing = np.random.uniform(0.6, 1.0)
        system_redundancy = np.random.uniform(0.8, 1.0)
        pilot_training_and_response = np.random.uniform(0.4, 1.0)
        passenger_safety_during_emergency = np.random.uniform(0.0, 1.0)
        communication_and_ground_support = np.random.uniform(0.7, 1.0)
        cabin_alertness = np.random.uniform(0.2, 1.0)

        # Example Feature: Turbulence Intensity
        # Higher turbulence intensity might lead to increased flight problems
        turbulence_intensity = np.random.uniform(0, 1)

        # Engineer Features
        # Engineer Features with Variation
        Engine_Performance_Feature = int(machine_speed >= machine_speed_tolerance * 0.8 and fuel_efficiency >= fuel_efficiency_min * 1.2)
        Battery_Health_Feature = int(battery_health >= battery_health_min * 0.9)
        Flight_Efficiency_Feature = int(time_savings >= time_savings_min * 1.1 and operational_costs <= operational_cost_max * 0.9)
        Emergency_Signals_Feature = int(emergency_signals >= emergency_signals_alert * 0.7)
        Weather_Impact_Feature = int(weather_impact >= 0.6)
        Terrain_Analysis_Feature = int(terrain_analysis >= 0.4)
        Safety_Features = int(passenger_safety_during_emergency >= 0.7 and communication_and_ground_support >= 0.8)

        Mechanical_Warning_Feature = int(part_temperature >= parts_temp_tolerance_max * 1.1)
        Cost_Inefficiencies_Warning_Feature = int(operational_costs >= operational_cost_max * 1.2)
        Emergency_Signals_Warning_Feature = int(emergency_signals >= emergency_signals_alert * 0.9)

        General_Warning_Feature = int(Mechanical_Warning_Feature or Cost_Inefficiencies_Warning_Feature or Emergency_Signals_Warning_Feature)

        flight_problems = int(General_Warning_Feature or flight_problems)

        data["time_savings"].append(time_savings)
        data["emergency_signals"].append(emergency_signals)
        data["fuel_efficiency"].append(fuel_efficiency)
        data["emission_reduction"].append(emission_reduction)
        data["part_temperature"].append(part_temperature)
        data["machine_speed"].append(machine_speed)
        data["measurement_accuracy"].append(measurement_accuracy)
        data["battery_health"].append(battery_health)
        data["motor_efficiency"].append(motor_efficiency)
        data["flight_time"].append(flight_time)
        data["flight_range"].append(flight_range)
        data["noise_level"].append(noise_level)
        data["charging_time"].append(charging_time)
        data["maintenance_intervals"].append(maintenance_intervals)
        data["avionics_performance"].append(avionics_performance)
        data["component_reliability"].append(component_reliability)
        data["energy_consumption"].append(energy_consumption)
        data["regulatory_compliance"].append(regulatory_compliance)
        data["passenger_load_factors"].append(passenger_load_factors)
        data["weather_impact"].append(weather_impact)
        data["material_usage_efficiency"].append(material_usage_efficiency)
        data["carbon_emissions"].append(carbon_emissions)
        data["operational_costs"].append(operational_costs)
        data["passenger_count"].append(passenger_count)
        data["time_savings_as_compared_driving"].append(time_savings_as_compared_driving)
        data["payload_capacity"].append(payload_capacity)
        data["emergency_landing_success_rate"].append(emergency_landing_success_rate)
        data["time_to_emergency_landing"].append(time_to_emergency_landing)
        data["safety_margin_for_emergency_landing"].append(safety_margin_for_emergency_landing)
        data["terrain_analysis"].append(terrain_analysis)
        data["weather_impact_on_emergency_landing"].append(weather_impact_on_emergency_landing)
        data["system_redundancy"].append(system_redundancy)
        data["pilot_training_and_response"].append(pilot_training_and_response)
        data["passenger_safety_during_emergency"].append(passenger_safety_during_emergency)
        data["communication_and_ground_support"].append(communication_and_ground_support)
        data["cabin_alertness"].append(cabin_alertness)

        data["Engine_Performance_Feature"].append(Engine_Performance_Feature)
        data["Battery_Health_Feature"].append(Battery_Health_Feature)
        data["Flight_Efficiency_Feature"].append(Flight_Efficiency_Feature)
        data["Emergency_Signals_Feature"].append(Emergency_Signals_Feature)
        data["Weather_Impact_Feature"].append(Weather_Impact_Feature)
        data["Terrain_Analysis_Feature"].append(Terrain_Analysis_Feature)
        data["Safety_Features"].append(Safety_Features)
        data["Mechanical_Warning_Feature"].append(Mechanical_Warning_Feature)
        data["Cost_Inefficiencies_Warning_Feature"].append(Cost_Inefficiencies_Warning_Feature)
        data["Emergency_Signals_Warning_Feature"].append(Emergency_Signals_Warning_Feature)
        data["General_Warning_Feature"].append(General_Warning_Feature)
        data["flight_problems"].append(flight_problems)

        # Example Feature: Turbulence Intensity
        # Higher turbulence intensity might lead to increased flight problems
        data["turbulence_intensity"].append(turbulence_intensity)

    return pd.DataFrame(data)


def main():
    features = [
        "Engine_Performance_Feature",
        "Battery_Health_Feature",
        "Flight_Efficiency_Feature",
        "Emergency_Signals_Feature",
        "Weather_Impact_Feature",
        "Terrain_Analysis_Feature",
        "Safety_Features",
        "Mechanical_Warning_Feature",
        "Cost_Inefficiencies_Warning_Feature",
        "Emergency_Signals_Warning_Feature",
        "General_Warning_Feature",
        "turbulence_intensity"
    ]
    #features = ["General_Warning_Feature", "Cost_Inefficiencies_Warning_Feature", "Emergency_Signals_Warning_Feature"]
    target_column = "flight_problems"
    num_data_points = 100000  # Number of data points for training
    num_new_data_points = 1000  # Number of new data points

    # Generate synthetic training data
    training_data = generate_synthetic_data(num_data_points)
    print("Training Data: ", training_data)

    # Train the RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=num_data_points, random_state=42)
    model.fit(training_data[features], training_data[target_column])

    # Visualize the data and results
    plot_histograms(training_data)
    plot_correlation_matrix(training_data)
    plot_feature_importance(model, features)
    plot_scatter('turbulence_intensity', target_column, training_data)

    # Generate new synthetic data for prediction
    new_data = generate_synthetic_data(num_new_data_points)

    # Make predictions on the new data using the trained model
    new_data_features = new_data[features]
    predictions = model.predict(new_data_features)

    # Print the predictions
    print("Predictions on New Data:", predictions)

    # Check the model training
    training_predictions = model.predict(training_data[features])
    training_accuracy = accuracy_score(training_data[target_column], training_predictions)
    print("Accuracy on Training Data:", training_accuracy)
    
    feature_importances = model.feature_importances_
    print("Feature Importances:", feature_importances)



if __name__ == "__main__":
    main()