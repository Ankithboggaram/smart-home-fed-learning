import polars as pl
import dask.dataframe as dd
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(
    input_csv: str,
    output_dir: str = 'data/',
    TRAIN_TEST_CUTOFF: str = '1451977136',
    TRAIN_VALID_RATIO: float = 0.75
    ) -> None:
    """
    Preprocesses the dataset by removing unecessarycolumns and scaling numeric values

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file to be split.

    Returns
    -------
    None
        The function saves one CSV file in the specified directory.

    """

    numeric_columns = [
    "use [kW]",
    "Furnace 1 [kW]",
    "Furnace 2 [kW]",
    "Home office [kW]",
    "Wine cellar [kW]",
    "Garage door [kW]",
    "Kitchen 12 [kW]",
    "Kitchen 14 [kW]",
    "Kitchen 38 [kW]",
    "Barn [kW]",
    "Living room [kW]",
    "Solar [kW]",
    "temperature",
    "humidity",
    "visibility",
    "apparentTemperature",
    "pressure",
    "windSpeed",
    "windBearing",
    "precipIntensity",
    "dewPoint",
    "precipProbability",
    ]
    target_Col = "Target"

    filepath = os.path.join(input_csv)
    df = pd.read_csv(filepath, index_col="time", dtype={"time": "str"})

    df = df.loc[:, numeric_columns]
    cols = df.columns

    df[target_Col] = df["use [kW]"].pct_change() * 100
    df[target_Col] = df["Target"].fillna(0)
    df = df.dropna()

    index = df.index[df.index < TRAIN_TEST_CUTOFF]
    index = index[: int(len(index) * TRAIN_VALID_RATIO)]
    scaler = StandardScaler().fit(df.loc[index, cols])
    df[cols] = scaler.transform(df[cols])

    filePathPreProcessed = os.path.join(output_dir, "SmartHomeDataset-Pre-Processed.csv")
    df.to_csv(filePathPreProcessed)





def split_train_test(
    input_csv: str,
    frac: float = 0.95,
    output_dir: str = 'data/'
    ) -> None:
    """
    Splits a dataset into training and test sets and saves them as CSV files.

    The function assumes that the dataset contains particular column names.
    This is not a general purpose function.

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
    df = pl.read_csv(input_csv, infer_schema_length=10000,
                     schema_overrides={
                            "time": pl.Utf8,
                            "cloudCover": pl.Utf8
                    })
    
    train_df = df.sample(fraction=frac)
    test_df = df.filter(~df["time"].is_in(train_df["time"]))

    train_path = output_dir + "train_data.csv"
    test_path = output_dir + "test_data.csv"

    train_df.write_csv(train_path)
    test_df.write_csv(test_path)

    print(f"Successfully split the dataset")


def split_csv_round_robin(
    input_csv: str,
    num_splits: int = 5,
    output_folder: str = 'data/split_data'
    ) -> None:
    """
    Splits the input CSV file into `num_splits` parts using a round-robin strategy and saves them as CSV files.

    The function assumes that the dataset contains particular column names.
    This is not a general purpose function.

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

    # Load the CSV file as a Dask DataFrame
    # df = dd.read_csv(input_csv, dtype={
    #                     "time": 'object',  # Similar to Utf8 in polars
    #                     "cloudCover": 'object',
    #                     'windBearing': 'float64'
    #                 })
    # df = dd.read_csv(input_csv)
    # Using pandas instead
    df = pd.read_csv(input_csv)

    df = df.assign(split_idx=(df.index % num_splits))

    for i in range(num_splits):
        split_df = df[df['split_idx'] == i]
        output_file = os.path.join(output_folder, f"home_{i + 1}.csv")
        split_df = split_df.drop(columns=['split_idx'])  # Remove the helper column
        # split_df.to_csv(output_file, single_file=True, index=False)
        split_df.to_csv(output_file, index=False)
        print(f"Saved {output_file}")


if __name__ =="__main__":

    # Preprocess the SmartHomeDataset
    preprocess_data('data/SmartHomeDataset.csv', output_dir='data')

    # Splitting dataset to train and test datasets
    split_train_test('data/SmartHomeDataset-Pre-Processed.csv', frac=0.95, output_dir='data/')

    # Round robin sampling of train_data into 5 splits
    split_csv_round_robin('data/train_data.csv')

