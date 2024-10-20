import polars as pl
import os



def split_train_test(input_csv: str, frac: float = 0.95, output_dir: str = 'data/') -> None:
    """
    Splits a dataset into training and test sets and saves them as CSV files.

    The function assumes that the dataset contains a column named 'time', which is used for splitting.
    If 'time' is not present, modify the column name to the appropriate identifier.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file to be split.
    frac : float, optional, default=0.95
        The fraction of data that will be used as the training set.
    output_dir : str, optional, default='data/'
        The directory where the output train and test CSV files will be saved.
        
    Returns
    -------
    None
        The function saves two CSV files: 'train_data.csv' and 'test_data.csv' in the specified directory.

    Example
    -------
    >>> split_train_test('input_dataset.csv', frac=0.8, output_dir='output/')
    """

    os.makedirs(output_dir, exist_ok=True)

    # It seems pl.Utf8 helps improve performance rather thatn just using str
    df = pl.read_csv(input_csv, dtypes={0: pl.Utf8, 27: pl.Utf8})
    
    train_df = df.sample(fraction=frac)
    test_df = df.filter(~df["time"].is_in(train_df["time"]))

    train_path = output_dir + "train_data.csv"
    test_path = output_dir + "test_data.csv"

    train_df.write_csv(train_path)
    test_df.write_csv(test_path)


def split_csv_round_robin(input_csv: str, num_splits: int = 5, output_folder: str = 'data/split_data') -> None:
    """
    Splits the input CSV file into `num_splits` parts using a round-robin strategy and saves them as CSV files.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file to be split.
    num_splits : int, optional, default=5
        The number of output splits.
    output_folder : str, optional, default='data/split_data'
        The directory where the output CSV files will be saved.
    
    Returns
    -------
    None
        The function saves multiple CSV files (one per split) in the specified directory.
    """
    
    os.makedirs(output_folder, exist_ok=True)

    df = pl.read_csv(input_csv, dtypes={0: pl.Utf8, 27: pl.Utf8})

    dfs = [pl.DataFrame() for _ in range(num_splits)]

    for i in range(len(df)):
        dfs[i % num_splits] = pl.concat([dfs[i % num_splits], df[i:i+1]])

    for i in range(num_splits):
        output_file = os.path.join(output_folder, f"home_{i + 1}.csv")
        dfs[i].write_csv(output_file)
        print(f"Saved {output_file}")


if __name__ =="__main__":

# split_csv_round_robin('data/HomeC.csv', num_splits=5, output_folder='data/split_data')