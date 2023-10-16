import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# import time
# import tkinter as tk

# # Create a progress bar
# progress_bar = tk.ProgressBar()
# progress_bar.pack()

# # Start the progress bar
# progress_bar.start()
# # Perform a long-running task
# for i in range(100):
#     # Update the progress bar
#     progress_bar.update()

#     # Wait for a short time
#     time.sleep(0.1)

# # Stop the progress bar
# progress_bar.stop()



class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

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

# Create the neural network model
model = NeuralNetwork()

# Customer data
num_customers = 1000
customer_data = pd.DataFrame({
'age': np.random.normal(40, 10, num_customers),
'satisfaction': np.random.randint(1, 10, num_customers),
'flight_history': np.random.poisson(5, num_customers),
})

# Bookings timeseries
num_days = 365 * 2
dates = pd.date_range('2020-01-01', periods=num_days)
bookings = pd.DataFrame({
'passengers': np.random.poisson(950, num_days),
'price': np.random.normal(250, 50, num_days),
}, index=dates)

# Air traffic data
num_flights = 365 * 500
flights = pd.DataFrame({
'airline': np.random.choice(['UA', 'AA', 'DL'], num_flights),
'origin': np.random.choice(['SFO', 'LAX', 'ORD', 'JFK'], num_flights),
'destination': np.random.choice(['SFO', 'LAX', 'ORD', 'JFK'], num_flights),
'departure_time': np.random.randint(0, 24, num_flights),
'flight_path': [np.random.choice(['A', 'B', 'C'], 10) for _ in range(num_flights)],
'runway': np.random.choice([1, 2, 3, 4], num_flights),
'landing_time': np.random.randint(0, 24, num_flights)
})

# Combine the customer data, bookings data, and flight data into a single DataFrame
data = pd.concat([customer_data, bookings, flights], axis=1)

# Split the data into training, validation, and test sets
print("training")
X_train, X_test, y_train, y_test = train_test_split(data, data['satisfaction'], test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
print("training end")

# Save the training, validation, and test sets
print("saving trained models")
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Define the loss function
def loss_fn(y_pred, y_true):
    """Calculates the loss function."""

    # Convert the Pandas Series to NumPy arrays
    y_true_numpy = y_true.values

    # Calculate the prediction of the neural network
    y_pred_numpy = y_pred().detach()

    # Convert the prediction to a tensor
    y_pred_tensor = torch.from_numpy(y_pred_numpy)

    # Calculate the loss
    loss = torch.mean((y_pred_tensor - y_true_numpy)**2)

    return loss
    
# class neural_network(nn.Module):
#     def __init__(self):
#         super(neural_network, self).__init__()

#         # Define the neural network architecture

#         self.fc1 = nn.Linear(12, 1)

#     def forward(self, x):
#         # Forward pass

#         x = torch.tanh(self.fc1(x))

#         return x

#     def predict(self, x):
#         # Predict the customer satisfaction

#         y_pred = self.forward(x)

#         return y_pred

# # Define the neural network architecture
# def neural_network(x):
#     if not torch.is_tensor(x):
#         raise ValueError('The input `x` must be a tensor.')
#     if x.size(0) > 0:
#         x = torch.tanh(torch.matmul(x.view(1, 12), torch.tensor([[1, 2]])))
#         x = torch.tanh(torch.matmul(x.view(1, 12), torch.tensor([[3, 4]])))
#         x = torch.sigmoid(torch.matmul(x, torch.tensor([[5, 6]])))
#     else:
#         # Handle empty matrix case
#         x = torch.zeros(1, 1)

#     return x

# Create the neural network model
#model = neural_network

# Train the neural network model
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100

for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_train)

    # Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss
    if epoch % 10 == 0:
        print('Epoch {}: Loss {}'.format(epoch, loss.item()))

# Evaluate the neural network model on the validation set
y_pred_val = model(X_val)
loss_val = loss_fn(y_pred_val, y_val)
print('Validation loss: {}'.format(loss_val.item()))

# Save the neural network model
torch.save(model.state_dict(), 'model.pt')

# Use the neural network model to predict customer satisfaction
customer_data = pd.read_csv('customer_data.csv')

# Predict the customer satisfaction for each customer
customer_satisfaction = model.predict(customer_data)

# Identify the customers who are at risk of churning
churn_risk = customer_satisfaction < 5

# Print the names of the customers who are at risk of churning
print(customer_data[churn_risk]['name'].tolist())
