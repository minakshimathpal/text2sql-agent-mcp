import pandas as pd
from io import StringIO
def markdown_to_df(markdown_table):
    data_io = StringIO(markdown_table)
    df = pd.read_csv(data_io, sep='|', skiprows=[2], skipinitialspace=True)
    df = df.iloc[1:, 1:-1]  # Remove first and last empty columns
    df.columns = [col.strip() for col in df.columns]  # Clean column names