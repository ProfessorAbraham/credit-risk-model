import unittest
import pandas as pd
from src.data_processing import calculate_rfm

class TestDataProcessing(unittest.TestCase):

    def test_calculate_rfm(self):
        # Sample transactions
        data = {
            'CustomerId': ['C1', 'C1', 'C2'],
            'TransactionStartTime': ['2025-06-01', '2025-06-10', '2025-06-05'],
            'TransactionId': [1, 2, 3],
            'Value': [100, 150, 200]
        }
        df = pd.DataFrame(data)
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        rfm = calculate_rfm(df, snapshot_date='2025-06-15')

        # Check if CustomerId count matches
        self.assertEqual(len(rfm), 2)

        # Check Recency for C1 is 5 days (15-10)
        self.assertEqual(rfm.loc[rfm['CustomerId'] == 'C1', 'Recency'].values[0], 5)

        # Check Frequency for C2 is 1
        self.assertEqual(rfm.loc[rfm['CustomerId'] == 'C2', 'Frequency'].values[0], 1)

if __name__ == '__main__':
    unittest.main()
