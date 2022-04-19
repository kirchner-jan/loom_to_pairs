from torch.utils.data import Dataset
import pickle

# It takes a file with loom embeddings and returns a dataset
class LoomComparisonsDataset(Dataset):
    def __init__(self , pickle_file):
        self.data = pickle.load(open(pickle_file, 'rb'))
    def __len__(self):
      return len(self.data)
    def __getitem__(self, idx):
      return self.data[idx]