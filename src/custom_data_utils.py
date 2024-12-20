import torch
from torch.utils.data import Dataset
import dask.dataframe as dd
import numpy as np


class SmartHomeDataset(Dataset):
    """
    A PyTorch dataset class for time series forecasting.

    This class expects the input data as sequences (e.g., sliding windows) from a time series and the corresponding target values.
    It is derived from `torch.utils.data.Dataset` and can be loaded to a `torch.utils.data.DataLoader`.

    Parameters
    -----------
    data_input: torch.FloatTensor
        Input sequences (sliding windows of time series)
    data_target: torch.FloatTensor
        Target values (e.g., future values to predict)
    input sequence_length int
        The length of each input sequence
    """

    def __init__(
        self,
        data_input: torch.FloatTensor,
        data_target: torch.FloatTensor,
        input_sequence_length: int,
    ):

        self.data_input = data_input
        self.data_target = data_target
        self.input_sequence_length = input_sequence_length

    def __len__(self):
        """Returns the number of sequences in the dataset."""
        return len(self.data_input)

    def __getitem__(self, idx):
        """
        Returns the input sequence and corresponding target value for the given index.

        Parameters
        ----------
        idx: int
            index of the sequence

        :return: a tuple of (input_sequence, target_value)
        """
        input_seq = self.data_input[idx]
        output_target = self.data_target[idx]
        return input_seq, output_target


def individual_home_datasets(
    # paths: list,  # Paths to the csv files containing data. Each path should be pointing at one dataset file in .csv format, e.g., ['data1_path', 'data2_path']  etc...
    inp_seq_len: int,
    # out_seq_len: int
) -> list[Dataset]:
    """Return a list of datasets that have been sampled by round robin sampling"""

    paths = [
        "../data/split_data/home_1.csv",
        "../data/split_data/home_2.csv",
        "../data/split_data/home_3.csv",
        "../data/split_data/home_4.csv",
    ]

    train_datasets = []

    for path in paths:

        # Read the CSV file and compute the Dask DataFrame
        df = dd.read_csv(path).compute()
        use_series = df.values

        # Normalize or standardize your features if necessary
        # Example: use_series = (use_series - np.mean(use_series)) / np.std(use_series)

        # Prepare input-output pairs
        data_input = []
        data_target = []
        for i in range(0, len(use_series) - inp_seq_len, inp_seq_len):
            data_input.append(use_series[i : i + inp_seq_len, :-1])
            data_target.append(use_series[i + inp_seq_len, -1])

        data_input = torch.FloatTensor(np.array(data_input))
        data_target = torch.FloatTensor(np.array(data_target))

        dataset = SmartHomeDataset(data_input, data_target, inp_seq_len)
        train_datasets.append(dataset)

    return train_datasets


def getDataset(
    type: str,
    inp_seq_len: int,
) -> Dataset:
    torch.manual_seed(42)

    file_path = "./data/test_data.csv"
    if type == "train":
        file_path = "./data/split_data/home_5.csv"

    df = dd.read_csv(file_path).compute()
    use_series = df.values

    data_input = []
    data_target = []

    for i in range(0, len(use_series) - inp_seq_len, inp_seq_len):
        data_input.append(use_series[i : i + inp_seq_len, :-1])
        data_target.append(use_series[i + inp_seq_len, -1])

    data_input = torch.FloatTensor(np.array(data_input))
    data_target = torch.FloatTensor(np.array(data_target))

    dataset = SmartHomeDataset(data_input, data_target, inp_seq_len)

    return dataset


def distribute_datasets(
    num_clients: int, client_id: int, inp_seq_len: int = 18, **kwargs
):
    """
    Return the smart home dataset for a given client.
    :param num_clients: total number of clients
    :param client_id: the client id
    """
    train_datasets = individual_home_datasets(inp_seq_len=18)

    # Common Test dataset for all clients
    # Read the CSV file and compute the Dask DataFrame
    df = dd.read_csv(
        "/home/ankith/github/smart-home-fed-learning/data/test_data.csv"
    ).compute()

    use_series = df.values

    # Normalize or standardize your features if necessary
    # Example: use_series = (use_series - np.mean(use_series)) / np.std(use_series)

    # Prepare input-output pairs
    data_input = []
    data_target = []
    for i in range(0, len(use_series) - inp_seq_len, inp_seq_len):
        data_input.append(use_series[i : i + inp_seq_len, :-1])
        data_target.append(use_series[i + inp_seq_len, -1])

    data_input = torch.FloatTensor(np.array(data_input))
    data_target = torch.FloatTensor(np.array(data_target))

    test_dataset = SmartHomeDataset(data_input, data_target, inp_seq_len)

    return train_datasets[client_id], test_dataset
