import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class NeuralNetwork(nn.Module):
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

def loss_fn(y_pred, y_true):
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

def load_data():
    # Load your data here and return the dataframes
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

    return customer_data, bookings, flights

def preprocess_data(customer_data, bookings, flights):
    # Combine the customer data, bookings data, and flight data into a single DataFrame
    data = pd.concat([customer_data, bookings, flights], axis=1)

    cust_tensor = torch.tensor(customer_data.values)
    book_tensor = torch.tensor(bookings.values) 
    flight_tensor = torch.tensor(flights.drop(columns=['airline', 'flight_path']).values)

    return cust_tensor, book_tensor, flight_tensor

# Concatenation
def concat_tensors(list_of_tensors):
   return torch.cat(list_of_tensors, dim=1)
   
# Train/test split 
def split_tensor(tensor, fractions):
   return torch.split(tensor, [int(x*len(tensor)) for x in fractions])

def split_data(data_tensor):
    # Split the data into training, validation, and test sets
    print("Split the data into training, validation, and test sets")

    X_train, X_test, y_train, y_test = train_test_split(data_tensor, data_tensor['satisfaction'], test_size=0.25, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    print("split complete")

    # Save the training, validation, and test sets
    need_files = False
    if need_files:
        print("Saving csv training, validation, and test files")
        X_train.to_csv('X_train.csv', index=False)
        X_val.to_csv('X_val.csv', index=False)
        X_test.to_csv('X_test.csv', index=False)
        y_train.to_csv('y_train.csv', index=False)
        y_val.to_csv('y_val.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)
        print("data sets saved")

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(model, X_train_tensor, y_train_tensor, epochs):
#def train_model(model, X_train, y_train, X_val, y_val, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert pandas DataFrames to PyTorch tensors
#    X_train_tensor = torch.tensor(X_train.values).float()
#    X_train_tensor = [torch.tensor(x) for x in X_train.values]
    # X_train_clean = np.array([x for x in X_train if isinstance(x, (int, float))])
    # X_train_tensor = [torch.tensor(x) for x in X_train_clean]
    # X_train_tensor = torch.tensor(X_train_tensor)
    X_train_tensor = torch.tensor(X_train_tensor.drop(columns=['airline']).values).float()

#    y_train_tensor = torch.tensor(y_train.values).float()
#    y_train_tensor = [torch.tensor(x) for x in y_train.values]
    y_train_tensor = torch.tensor(y_train_tensor.values).float()

#    X_val_tensor = torch.tensor(X_val.values).float()
    #X_val_tensor = [torch.tensor(x) for x in X_val_tensor.values]
    X_val_clean = np.array([x for x in X_val_tensor if isinstance(x, (int, float))])
    X_val_tensor = [torch.tensor(x) for x in X_val_clean]

#    y_val_tensor = torch.tensor(y_val.values).float()
    y_val_tensor = [torch.tensor(x) for x in y_val.values]

    for epoch in tqdm(range(epochs)):
        # Forward pass
#        y_pred = model(X_train_tensor)
        y_pred = model(X_train_tensor.unsqueeze(0))

        # Calculate the loss
        loss = loss_fn(y_pred, y_train_tensor)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate the model on the validation set
        y_pred_val = model(X_val_tensor)
        loss_val = loss_fn(y_pred_val, y_val_tensor)

        # Save and Print the loss
        if epoch % 10 == 0:
            print('Epoch {}: Training loss: {} Validation loss: {}'.format(epoch, loss.item(), loss_val.item()))
            torch.save(model.state_dict(), 'model_{}.pt'.format(epoch))

    # Save the final model
    return torch.save(model.state_dict(), 'sac-sat-model.pt')

# def train_model(model, X_train, y_train, X_val, y_val, epochs):
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     for epoch in tqdm(range(epochs)):
#         # Forward pass
#         y_pred = model(X_train)

#         # Calculate the loss
#         loss = loss_fn(y_pred, y_train)

#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Evaluate the model on the validation set
#         y_pred_val = model(X_val)
#         loss_val = loss_fn(y_pred_val, y_val)

#         # Save and Print the loss
#         if epoch % 10 == 0:
#             print('Epoch {}: Loss {}'.format(epoch, loss.item()))
#             print('Epoch {}: Training loss: {} Validation loss: {}'.format(epoch, loss.item(), loss_val.item()))
#             torch.save(model.state_dict(), 'model_{}.pt'.format(epoch))

#     # Save the final model
#     return torch.save(model.state_dict(), 'sac-sat-model.pt')

def evaluate_model(model, X_test, y_test):
    # Evaluate your model here and return the loss
    # Evaluate the model on the test set
    y_pred_test = model(X_test)
    loss_test = loss_fn(y_pred_test, y_test)

    # Print the loss
    print('Test loss: {}'.format(loss_test.item()))

    return loss_test.item()

def predict_churn(model, cust_tensor):
#def predict_churn(model, customer_data):
    # Predict customer satisfaction here and return the churn risk
#    customer_satisfaction = model.predict(customer_data)
    customer_satisfaction = model.predict(cust_tensor)

    # Identify the customers who are at risk of churning
    churn_risk = customer_satisfaction < 5

    # Print the names of the customers who are at risk of churning
#    print(customer_data[churn_risk]['name'].tolist())
    print(cust_tensor[churn_risk]['name'].tolist())

    return churn_risk

def main():
    customer_data, bookings, flights = load_data()
    
    data = preprocess_data(customer_data, bookings, flights)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    
    model = NeuralNetwork()
    
    train_model(model, X_train, y_train, X_val, y_val, 100)
    
    test_loss = evaluate_model(model, X_test, y_test)
    
    print('Test loss: {}'.format(test_loss))

if __name__ == "__main__":
    main()
