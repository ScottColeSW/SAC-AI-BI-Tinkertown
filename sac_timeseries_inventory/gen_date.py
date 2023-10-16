import random
import datetime
import calendar

def get_bounded_random_date(start_date, end_date):
  """
  Gets a random date between start_date and end_date.

  Args:
    start_date: The start date.
    end_date: The end date.

  Returns:
    A random date between start_date and end_date.
  """

  random_year = random.randint(start_date.year, end_date.year)
  random_month = random.randint(start_date.month, end_date.month)
#  random_day = random.randint(start_date.day, end_date.day)
  random_day = random.randint(1, calendar.monthrange(random_year, random_month)[1])

  return datetime.date(random_year, random_month, random_day)

#print(get_bounded_random_date(datetime.date(2000, 1, 1), datetime.date(2023, 12, 31)))
