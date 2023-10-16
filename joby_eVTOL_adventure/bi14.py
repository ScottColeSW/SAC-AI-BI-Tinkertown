import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configs
DAYS_PER_YEAR = 365
TIMESTEPS = 2 # 2 years
FLIGHTS_DAILY = 500
TRAINING_EPOCHS = 250
NUM_CUSTOMERS = FLIGHTS_DAILY * TIMESTEPS * DAYS_PER_YEAR

def create_load_data(num_customers, time_step, flights_daily):
    # Customer data
    customer_data = pd.DataFrame({
        'age': np.random.normal(40, 10, num_customers),
        'satisfaction': np.random.randint(1, 10, num_customers),
        'flight_history': np.random.poisson(5, num_customers),
    })

    # Bookings timeseries
    num_days = DAYS_PER_YEAR * time_step
    dates = pd.date_range('2024-01-01', periods=num_days)
    bookings = pd.DataFrame({
        'passengers': np.random.poisson(950, num_days),
        'price': np.random.normal(250, 50, num_days),
    }, index=dates)

    # Air traffic data
    num_flights = DAYS_PER_YEAR * flights_daily
    flights = pd.DataFrame({
        'airline': np.random.choice(['UA', 'AA', 'DL'], num_flights),
        'origin': np.random.choice(['SFO', 'LAX', 'ORD', 'JFK'], num_flights),
        'destination': np.random.choice(['SFO', 'LAX', 'ORD', 'JFK'], num_flights),
        'departure_time': np.random.randint(0, 24, num_flights),
        'flight_path': [np.random.choice(['A', 'B', 'C'], 10) for _ in range(num_flights)],
        'runway': np.random.choice([1, 2, 3, 4], num_flights),
        'landing_time': np.random.randint(0, 24, num_flights)
    })

    return customer_data, bookings, flights

# Data simulation
def simulate_data():
    cust_df, book_df, flight_df = create_load_data(NUM_CUSTOMERS, TIMESTEPS, FLIGHTS_DAILY)

    return cust_df, book_df, flight_df

# Analytics model
def analytics_model(cust_df, book_df, flight_df):
    # Simple aggregation, summary statistics
    # TODO fill in the analytics that produce the same kinds of results as the AI
    print("Average Customer Satisfaction:", cust_df['satisfaction'].mean())
    print("Total Revenue:", book_df['price'].sum() * book_df['passengers'].sum())
    print("Average Flight Delay:", flight_df['landing_time'].sub(flight_df['departure_time']).mean())

    return

def train_and_evaluate(AIModel, data_df, target_column, validation_split=0.25):
#   """Trains and evaluates an AI model on a held-out test set.

#   Args:
#     AIModel: The AI model to train and evaluate.
#     data_df: The data frame to train and evaluate the model on.
#     target_column: The name of the target column in the data frame.
#     validation_split: The proportion of the data to hold out for validation.

#   Returns:
#     The RMSE of the AI model on the held-out test set.
#   """

    # Split the data into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        data_df, data_df[target_column], test_size=validation_split)

    # Train the AI model.
    model = AIModel()
    best_model = None
    best_rmse = np.inf
    for epoch in range(TRAINING_EPOCHS):
        model.fit(X_train, y_train)

        # Evaluate the AI model on the validation set.
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_pred - y_test)**2))

        # Save the model if it has the best RMSE so far.
        if rmse < best_rmse:
            best_model = model
            best_rmse = rmse
            # Save and Print the loss
            if epoch % 10 == 0:
                print('Best Model:\n{best_model}\nBest RMSE:\n{}', best_model, best_rmse)
                torch.save(model.state_dict(), 'model_{}.pt'.format(epoch))

    # Return the RMSE of the best model on the held-out test set.
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(np.mean((y_pred - y_test)**2))
    return rmse


# AI model - Neural network architecture
class AIModel(nn.Module):
    def __init__(self):
        super(AIModel, self).__init__()

        # Define the hidden layers
        self.hidden1 = nn.Linear(12, 12)
        self.hidden2 = nn.Linear(12, 12)

        # Define the output layer
        self.output = nn.Linear(12, 1)

    def forward(self, x):
        # Pass the input through the hidden layers
        x = nn.functional.tanh(self.hidden1(x))
        x = nn.functional.tanh(self.hidden2(x))

        # Pass the output through the sigmoid function
        x = nn.functional.sigmoid(self.output(x))
        return x

    def loss_fn(self, y_pred, y_true):
        #"""Calculates the loss function."""

        # Convert the Pandas Series to NumPy arrays
        y_true_numpy = y_true.values

        # Calculate the prediction of the neural network
        y_pred_numpy = y_pred().detach()

        # Convert the prediction to a tensor
        y_pred_tensor = torch.from_numpy(y_pred_numpy)

        # Calculate the loss
        loss = torch.mean((y_pred_tensor - y_true_numpy)**2)
        return loss

    def fit(self, X_train, y_train):
    # """Trains the model on the given training data.

    # Args:
    #     X_train: A tensor containing the training data.
    #     y_train: A tensor containing the training labels.
    # """

        # Set the model to training mode.
        self.train()

        # Optimize the model parameters.
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(100):
            optimizer.zero_grad()

            # Forward pass.
            y_pred = self(X_train)

            # Compute the loss.
            loss = self.loss_fn(y_pred, y_train)

            # Backward pass.
            loss.backward()

            # Update the model parameters.
            optimizer.step()

        # Set the model to evaluation mode.
        self.eval()

# Benchmark
cust_df, book_df, flight_df = simulate_data()

customer_satisfaction_rmse = train_and_evaluate(AIModel, cust_df, 'satisfaction')
flight_delays_rmse = train_and_evaluate(AIModel, flight_df, 'departure_time')

#ai_rmse = train_and_evaluate(AIModel(), cust_df, book_df, flight_df) 
baseline_rmse = analytics_model(cust_df, book_df, flight_df)

print(f'Dataframes:\nCustomers\n', cust_df, '\nBookings\n',book_df, '\nFlights\n', flight_df)
#print(f'AI RMSE: {ai_rmse}') 
print(f'Baseline RMSE: {baseline_rmse}')

print(f'AI customer_satisfaction_rmse: {customer_satisfaction_rmse}') 
print(f'AI flight_delays_rmse: {flight_delays_rmse}')