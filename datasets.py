import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import random
from constants import DEFAULT_MODULO
import itertools

def one_hot_encode(number, size):
    one_hot = torch.zeros(size)
    one_hot[number] = 1
    return one_hot

def binary_encode(num, bits):
    return torch.tensor([int(b) for b in format(num, f'0{bits}b')])

def generate_random_one_hot(length):
    index = torch.randint(0, length, (1,)).item()
    one_hot_vector = torch.zeros(length)
    one_hot_vector[index] = 1
    return one_hot_vector

def unique_random_combinations(num_features, num_samples):
    seen = set()
    domain = [0, 1]

    if num_samples> 2**num_features:
        print(f"Number of samples > Possible combinations, setting num_samples to {2**num_features}")
        num_samples = 2**num_features
    while len(seen) < num_samples:
        combination = tuple(random.choice(domain) for _ in range(num_features))
        if combination not in seen:
            seen.add(combination)
            yield combination


class AlgorithmicDataset(Dataset):
    def __init__(self, operation, p=DEFAULT_MODULO, input_size=None, output_size=None, one_hot=True):
        self.p = p
        if not input_size:
            self.input_size = p
        else:
            self.input_size = input_size
        self.output_size = output_size if output_size else self.p
        self.operation = operation
        
        self.data = []
        self.targets = []

        for x in range(0,self.p):
            for y in range(0,self.p):
                if 'div' in operation.__name__ and y == 0:
                    continue
                result = self.operation(x, y) % self.p 
                if one_hot:
                    x_one_hot = one_hot_encode(x, self.input_size)
                    y_one_hot = one_hot_encode(y, self.input_size)
                    combined_input = torch.cat((x_one_hot, y_one_hot), 0)
                else:
                    combined_input = torch.tensor((x, y), dtype=torch.long)

                self.data.append(combined_input)
                self.targets.append(result)

        self.data = torch.stack(self.data)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    

class BinaryAlgorithmicDataset(Dataset):
    def __init__(self, operation, p=DEFAULT_MODULO, input_size=None, output_size=None, shuffle=False):
        self.p = p
        if not input_size:
            self.input_size = p
        else:
            self.input_size = input_size
        self.output_size = output_size if output_size else self.p
        self.operation = operation
        self.shuffle = shuffle
        
        self.data = []
        self.targets = []

        input_bits = (self.input_size - 1).bit_length()
        output_bits = (self.output_size - 1).bit_length()

        if self.shuffle:
            indices = list(range(self.input_size))
            random.shuffle(indices)
            self.permutation_mapping = indices
        else:
            self.permutation_mapping = None

        for x in range(0, self.p):
            for y in range(0, self.p):
                if 'div' in operation.__name__ and y == 0:
                    continue
                result = self.operation(x, y) % self.p 

                if self.permutation_mapping is not None:
                    x_mapped = self.permutation_mapping[x]
                    y_mapped = self.permutation_mapping[y]
                else:
                    x_mapped = x
                    y_mapped = y

                x_binary = binary_encode(x_mapped, input_bits)
                y_binary = binary_encode(y_mapped, input_bits)
                result = one_hot_encode(result, self.output_size)

                combined_input = torch.cat((x_binary, y_binary), 0)
                self.data.append(combined_input)
                self.targets.append(result)

        self.data = torch.stack(self.data)
        self.targets = torch.stack(self.targets)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    

class SparseParityDataset(Dataset):
    def __init__(self, num_features, num_noise_features, num_samples=None):
        self.num_features = num_features
        self.num_noise_features = num_noise_features
        self.num_samples = num_samples
    
        self.data = torch.tensor(list(unique_random_combinations(num_features + self.num_noise_features, self.num_samples)))
        self.targets = (self.data[:,:num_features].sum(dim=1)%2).float()
        self.targets = self.targets.long()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]