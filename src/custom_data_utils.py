import torch
from torch.utils.data import Dataset
import dask.dataframe as dd

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
        input_sequence_length: int
        output_sequence_length: int
        ):

        self.data_input = data_input
        self.data_target = data_target
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

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
        input_seq = self.data_input[idx : idx + self.input_sequence_length]
        output_seq = self.data_target[idx + self.input_sequence_length : idx + self.input_sequence_length + self.output_sequence_length]
        return input_seq, output_seq


def individual_home_datasets(
    paths: List, # Paths to the csv files containing data. Each path should be pointing at one dataset file in .csv format, e.g., ['data1_path', 'data2_path']  etc...
    inp_seq_len: int,
    out_seq_len: int
) -> List[Dataset]:
    
    """Return a list of datasets that have been sampled by round robin sampling"""
    
    train_datasets = []
    
    for path in paths:
        df = dd.read_csv(path)
        train_sequence = []
        test_sequence =  []
        # TODO: Check how to split dataset into partitions for forecasting (Should be similar to the pytorch thing you've done earlier)
        for i in range(len(df)):

