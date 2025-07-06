from torch.utils.data import random_split, Dataset, DataLoader
import importlib
import tiktoken
import lightning as L
import torch

class GPTDataset(Dataset):
    def __init__(
        self,
        samples
    ):
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class GPTDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_path: str, 
        train_perc=0.65, 
        val_perc=0.15, 
        batch_size=32,
        max_length=4,
        stride=4
    ):
        super().__init__()
        self.data_path = data_path
        self.train_perc = train_perc
        self.val_perc = train_perc + val_perc
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride = stride

    # def prepare_data(self):
    #     # download data from web if needed, runs on single process
        

    def setup(self):
        # called after finishing prepare_data, sets up each GPU in multi-GPU setting
        tokenizer = tiktoken.get_encoding("gpt2")
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        token_ids = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > self.max_length, "Number of tokenized inputs must at least be equal to max_length+1"
        
        input_ids = []
        target_ids = []
        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - self.max_length, self.stride):
            input_chunk = token_ids[i:i + self.max_length]
            target_chunk = token_ids[i + 1: i + self.max_length + 1]
            input_ids.append(torch.tensor(input_chunk))
            target_ids.append(torch.tensor(target_chunk))

        n = len(input_ids)
        n_train = int(self.train_perc*n)
        n_eval = int(self.val_perc*n)
        self.train_data = GPTDataset(list(zip(input_ids[:n_train], target_ids[:n_train])))
        self.eval_data = GPTDataset(list(zip(input_ids[n_train:n_eval], target_ids[n_train:n_eval])))
        self.test_data = GPTDataset(list(zip(input_ids[n_train:n_eval], target_ids[n_train:n_eval])))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.eval_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


import unittest
class TestGPTDataModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 8
        cls.dm = GPTDataModule(data_path= "./the-verdict.txt", batch_size=cls.batch_size)
        cls.dm.setup()

    @classmethod
    def tearDownClass(cls):
        del cls.dm

    def test_setup(self):
        self.assertIsNotNone(self.dm.train_data)
        self.assertIsNotNone(self.dm.eval_data)
        self.assertIsNotNone(self.dm.test_data)
        self.assertEqual(len(self.dm.train_data), 835)
        self.assertEqual(len(self.dm.eval_data), 193)
        self.assertEqual(len(self.dm.test_data), 193)

    def test_dataloader_shape(self):
        for loader_fn in [self.dm.train_dataloader, self.dm.val_dataloader, self.dm.test_dataloader]:
            loader = loader_fn()
            x, y = next(iter(loader))
            self.assertEqual(x.shape, (self.batch_size, 4))
            self.assertEqual(y.shape, (self.batch_size, 4))
    

if __name__ == '__main__':
    unittest.main()