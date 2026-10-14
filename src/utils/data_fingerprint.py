import hashlib
import pandas as pd

def get_data_fingerprint(df):

    row_hashes = pd.util.hash_pandas_object(df, index=True).values
    data_md5 = hashlib.md5(row_hashes.tobytes()).hexdigest()[:10]

    return data_md5

if __name__ == "__main__":

    df = pd.read_csv('data/input_data.csv')
    print(get_data_fingerprint(df))
