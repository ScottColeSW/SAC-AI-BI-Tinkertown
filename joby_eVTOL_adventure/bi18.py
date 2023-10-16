import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Configs
DAYS_PER_YEAR = 365
TIMESTEPS = 2 # 2 years
FLIGHTS_DAILY = 50
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
        'passengers': np.random.poisson(1, num_days),
        'price': np.random.normal(25, 150, num_days),
    }, index=dates)

    # Air traffic data
    num_flights = DAYS_PER_YEAR * flights_daily
    flights = pd.DataFrame({
        'airline': np.random.randint(1, 3, num_flights),
        'origin': np.random.randint(1, 4, num_flights),
        'destination': np.random.randint(1, 4, num_flights),
        'departure_time': np.random.randint(0, 24, num_flights),
        'flight_path': np.random.randint(1, 3, num_flights),
        'runway': np.random.randint(1, 4, num_flights),
        'landing_time': np.random.randint(0, 24, num_flights)
    })

    # Convert the NumPy arrays to PyTorch tensors
    customer_data_tensor = torch.tensor(customer_data.values).float()
    bookings_tensor = torch.tensor(bookings.values).float()
    flights_tensor = torch.tensor(flights.values).float()

    return customer_data_tensor, bookings_tensor, flights_tensor, customer_data, bookings, flights

# Data simulation
def simulate_data():
    cust_tensor, book_tensor, flight_tensor, customer_data_df, bookings_df, flights_df = create_load_data(NUM_CUSTOMERS, TIMESTEPS, FLIGHTS_DAILY)

    return cust_tensor, book_tensor, flight_tensor, customer_data_df, bookings_df, flights_df

# Analytics model
def analytics_model(cust_df, book_df, flight_df):
    # Simple aggregation, summary statistics
    print("Average Customer Satisfaction:", cust_df['satisfaction'].mean())
    print("Total Revenue:", book_df['price'].sum() * book_df['passengers'].sum())
    print("Average Flight Delay:", flight_df['landing_time'].sub(flight_df['departure_time']).mean())

    return

def create_label_tensor(data_frame: pd.DataFrame, target_column: str) -> torch.Tensor:
    # Creates a label tensor to match with the data tensor.

    # Args:
    #     data_frame: A pandas DataFrame containing the data.
    #     target_column: The name of the target column in the data frame.

    # Returns:
    #     A tensor containing the labels.
# Check if the target column exists in the data frame.
    if target_column not in data_frame.columns:
        raise KeyError(f"The target column '{target_column}' does not exist in the data frame.")

    # Convert the target column to a tensor.
    label_tensor = torch.tensor(data_frame[target_column].values).float().view(-1,1)

    # # Get the index of the target column in the data frame.
    # target_column_index = data_frame.columns.get_loc(target_column)

    # # Convert the target column index to a one-dimensional integer tensor.
    # target_column_index_tensor = torch.tensor([target_column_index])

    # # Create a new tensor to store the labels.
    # label_tensor = torch.empty(data_frame.shape[0])

    # # Iterate over the data frame and extract the target column for each row.
    # for i in range(data_frame.shape[0]):
    #     label_tensor[i] = torch.index_select(data_frame[i], 0, target_column_index_tensor)

    # Return the label tensor.
    return label_tensor

def train_and_evaluate(AIModel, data_df, target_column, validation_split=0.25):
    #  Trains and evaluates an AI model on a held-out test set.

    #  Args:
    #  AIModel: The AI model to train and evaluate.
    #  data_df: The data frame to train and evaluate the model on.
    #  target_column: The name of the target column in the data frame.
    #  validation_split: The proportion of the data to hold out for validation.

    #  Returns:
    #  The RMSE of the AI model on the held-out test set.

    # Loss function 
    def loss_fn(y_pred, y_true):
        # PyTorch loss calculations
        loss = nn.MSELoss()(y_pred.float(), y_true.float())
        return loss
    
    # Evaluate function
    def evaluate(model, X, y):
        predictions = model(X.float())
        loss = loss_fn(predictions, y.float())
        return loss

    # Convert the DataFrame to a tensor
    data_tensor = torch.tensor(data_df.values)

    # Create a label tensor to match with the data tensor.
    label_tensor = create_label_tensor(data_df, target_column)

    # Split the data into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(data_tensor, label_tensor, test_size=validation_split, train_size=0.75, random_state=42)

    # Train the AI model.
    model = AIModel
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_model = None
    best_rmse = float('inf')
    debug = False

    for epoch in range(TRAINING_EPOCHS):
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X_train.float()) 
        
        # Calculate loss
        loss = loss_fn(y_pred.float(), y_train.float())
        
        # Backprop
        loss.backward()
        
        # Update weights
        optimizer.step()

        # Save the model if it has the best RMSE so far.
        if loss < best_rmse:
            best_model = model
            best_rmse = loss
            # Save and Print the loss
            if epoch % 100 == 0:
                if debug:
                    print('Best Model:\n', best_model)
                    print('Best RMSE:\n', best_rmse)
                #print('Best Model:\n{}\nBest RMSE:\n{}', best_model, best_rmse)
                    print('Epoch:\n',epoch)
                torch.save(model.state_dict(), 'model_{}.pt'.format(epoch))

    # Save final model
    torch.save(model, 'model_sav_sac{}.pt'.format(epoch))

    # Return the RMSE of the best model on the held-out test set.
    rmse = evaluate(best_model, X_test, y_test)
    return rmse

def create_linear_regression_model(X_train, y_train, X_val, y_val, X_test, y_test):
    # Create a linear regression model
    model = LinearRegression()

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_pred_val = model.predict(X_val)
    rmse_val = np.sqrt(np.mean((y_pred_val - y_val)**2))

    # Evaluate the model on the test set
    y_pred_test = model.predict(X_test)
    rmse_test = np.sqrt(np.mean((y_pred_test - y_test)**2))

    # Print the RMSE on the validation and test sets
    print('Validation RMSE:', rmse_val)
    print('Test RMSE:', rmse_test)

# AI model - Neural network architecture
# class AIModel(nn.Module):
#     def __init__(self):
#         super(AIModel, self).__init__()

#         # Define the hidden layers
#         self.hidden1 = nn.Linear(3, 12)  # Adjusted input shape from 12 to 3
#         self.hidden2 = nn.Linear(12, 12)

#         # Define the output layer
#         self.output = nn.Linear(12, 1)

#     def forward(self, x=None):
#         if x is None:
#             x = torch.randn(10, 12)
#         else:            
#             # Pass the input through the hidden layers
#             x = nn.functional.tanh(self.hidden1(x))
#             x = nn.functional.tanh(self.hidden2(x))

#             # Pass the output through the sigmoid function
#             x = torch.sigmoid(self.output(x))

#         return x

#     def loss_fn(self, y_pred, y_true):
#         #"""Calculates the loss function."""

#         # Convert the Pandas Series to NumPy arrays
#         y_true_numpy = y_true.values

#         # Calculate the prediction of the neural network
#         y_pred_numpy = y_pred().detach()

#         # Convert the prediction to a tensor
#         y_pred_tensor = torch.from_numpy(y_pred_numpy)

#         # Calculate the loss
#         loss = torch.mean((y_pred_tensor - y_true_numpy)**2)

#         return loss

#     def fit(self, X_train, y_train):
#     # """Trains the model on the given training data.

#     # Args:
#     #     X_train: A tensor containing the training data.
#     #     y_train: A tensor containing the training labels.
#     # """

#         # Set the model to training mode.
#         self.train()

#         # Optimize the model parameters.
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
#         for epoch in range(100):
#             optimizer.zero_grad()

#             # Forward pass.
#             y_pred = self(X_train)

#             # Compute the loss.
#             loss = self.loss_fn(y_pred, y_train)

#             # Backward pass.
#             loss.backward()

#             # Update the model parameters.
#             optimizer.step()

#         # Set the model to evaluation mode.
#         self.eval()


class AIModel(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Args:
            input_size: The size of the input layer.
            output_size: The size of the output layer.
        """

        super(AIModel, self).__init__()

        hidden_size = input_size // 2 

        self.output = nn.Linear(input_size, hidden_size)

        # # Define the hidden layers
        # if hidden_layers is None:
        #     hidden_layers = []

        # for i in range(len(hidden_layers)):
        #     input_size, output_size = hidden_layers[i]
        #     setattr(self, f"hidden{i}", nn.Linear(input_size, output_size))

        # # Define the output layer
        # if hidden_layers:
        #     self.output = nn.Linear(hidden_layers[-1][1], output_size)
        # else:
        #     self.output = nn.Linear(input_size, output_size)

    # def forward(self, x):
    #     # """
    #     # Args:
    #     #     x: A tensor containing the input data.

    #     # Returns:
    #     #     A tensor containing the output of the model.
    #     # """

    #     for i in range(len(self.hidden)):
    #         x = nn.functional.tanh(getattr(self, f"hidden{i}")(x))

    #     # Pass the output through the sigmoid function
    #     x = torch.sigmoid(self.output(x))

    #     return x

    def forward(self, x):
        return torch.sigmoid(self.output(x))

    def loss_fn(self, y_pred, y_true):
        # """
        # Calculates the loss function.

        # Args:
        #     y_pred: A tensor containing the predicted output.
        #     y_true: A tensor containing the true output.

        # Returns:
        #     A tensor containing the loss.
        # """

        # Calculate the loss
        loss = torch.mean((y_pred - y_true)**2)

        return loss

    def fit(self, X_train, y_train):
        # """
        # Trains the model on the given training data.

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

# model = AIModel(input_size=3, output_size=1, hidden_layers=[(3,2)])
model = AIModel(3,1)
X_trainx = torch.randn(1000, 3)
y_trainx = torch.randn(1000, 1)

model.fit(X_trainx, y_trainx)
X_new = torch.randn(10, 3)
y_pred = model(X_new)
print(f'y_pred = {y_pred}\n')

# Benchmark
cust_tensor, book_tensor, flight_tensor, cust_df, book_df, flight_df = simulate_data()

# if debug: print(f'bookings, flights, customers:\n{book_df}\n{flight_df}\n{cust_df}\n')

#print(f'Analytics Models: {cust_df}, {book_df}, {flight_df}\n')
analytics_model(cust_df, book_df, flight_df)

print("Getting Models and RSME results:\n")
# Train and evaluate the AI model on customer satisfaction
customer_satisfaction_rmse = train_and_evaluate(AIModel(3,1), cust_df, 'satisfaction')
# Print the RMSE
print(f'Customer Satisfaction RMSE: {customer_satisfaction_rmse}')
passenger_rmse = train_and_evaluate(AIModel(3,1), book_df, 'passengers')
print(f'Passenger RMSE: {passenger_rmse}')
# price_rmse = train_and_evaluate(AIModel(), book_df, 'price')
# print(f'Price RMSE: {price_rmse}')