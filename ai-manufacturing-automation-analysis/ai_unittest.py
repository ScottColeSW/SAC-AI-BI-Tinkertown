import unittest
import pandas as pd
import numpy as np
from ai_detect import generate_synthetic_data, identify_problems

class TestManufacturingAnalysis(unittest.TestCase):

    def test_generate_synthetic_data(self):
        num_points = 100
        data = generate_synthetic_data(num_points)

        self.assertIsInstance(data, pd.DataFrame)
        #print(f"data length: ", range(len(data)), "data shape: ", data.shape)
        #print("col: ", data.shape[1], "rows: ", )

        self.assertEqual(data.shape, (num_points, data.shape[1]))  # Adjust to match your data structure

    def test_identify_problems(self):
        # Create a sample dataframe for testing
        sample_data = {
            "part_temperature": [100, 110, 120, 130, 140],
            "machine_speed": [1000, 1100, 1200, 1300, 1400],
            "measurement_accuracy": [1, 2, 3, 4, 5]
        }
        dataframe = pd.DataFrame(sample_data)
        #print("Data shape:", dataframe.shape)  # Print the shape of the generated data

        threshold = 0.3
        problematic_rows = identify_problems(dataframe, threshold)

        self.assertIsInstance(problematic_rows, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()