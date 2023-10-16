# import random
# import datetime
# import gen_date as g
# import pandas as pd

# def data_simulation_generator(N):
#   """
#   Creates N rows of data for customer behavior analysis and prediction.

#   Args:
#     N: The number of rows of data to create.

#   Returns:
#     A DataFrame of N rows of data.
#   """

#   customer_ids = [random.randint(1, 10000) for i in range(N)]
#   dates = [g.get_bounded_random_date(datetime.date(2000, 1, 1), datetime.date(2023, 12, 31)) for i in range(N)]
#   #dates = [g.get_bounded_random_date(datetime.date(2000, 1, 1), datetime.date(2023, 12, 31))]
#   #dates = [random.date(2023, i, 1) for i in range(1, N + 1)]
#   product_ids = [random.randint(1, 1000) for i in range(N)]
#   quantities = [random.randint(1, 10) for i in range(N)]
#   stock = [random.randint(-10, 10) for i in range(N)]
#   backlog = [stock[i] < 0: True for i in range(N)]
#   prices = [random.randint(1, 100) for i in range(N)]

#   data = {
#     "Customer_ID": customer_ids,
#     "Date": dates,
#     "Product_ID": product_ids,
#     "Quantity": quantities,
#     "Stock": stock,
#     "Backlog": backlog,
#     "Price": prices
#   }

#   df = pd.DataFrame(data)

#   return df

# print(data_simulation_generator(105))




import random
import datetime
import numpy as np
import pandas as pd

def data_simulation_generator(N):
  """
  Creates N rows of data for customer behavior analysis and prediction.

  Args:
    N: The number of rows of data to create.

  Returns:
    A DataFrame of N rows of data.
  """


# Current inventory levels
# Outstanding purchase orders
# Forecasting period requirements
# Expected demand and seasonality
# Maximum possible stock levels
# Sales trends and velocity
# Customer response to specific products

  customer_ids = np.random.randint(1, 1000000, N)
  dates = np.random.randint(datetime.date(2000, 1, 1).toordinal(), datetime.date(2023, 12, 31).toordinal(), N)
  #  dates = np.random.randint(datetime.date(2000, 1, 1), datetime.date(2023, 12, 31), N)
  product_ids = np.random.randint(1, 100, N)
  quantities = np.random.randint(1, 10000, N)
  stock = np.random.randint(-100, 14000, N)
  on_order = quantities + stock
  backlog = stock < 0
  prices = np.random.randint(1, 10000, N)

  data = {
    "Customer_ID": customer_ids,
    "Date": dates,
    "Product_ID": product_ids,
    "Quantity": quantities,
    "Stock": stock,
    "Ordered": on_order,
    "Backlog": backlog,
    "Price": prices
  }

  df = pd.DataFrame(data)

  return df

print(data_simulation_generator(1005))
