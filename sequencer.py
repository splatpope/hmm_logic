import json
import os
from datetime import datetime
from typing import Callable, Iterable, List, Optional, Tuple
from random import shuffle
from math import floor

from numpy import array_split

Sequence = List[int]
Items_List = List[str]
Total_Indices = List[int]
Training_Indices = List[List[int]]
Testing_Indices = List[int]
Index_Split = Tuple[Testing_Indices, Training_Indices]

class Sequencer:

    savefile_loc = "./data/sequencers"

    def __init__(self, name: str) -> None:
        self.name = name
        self.compressed = False
        self.data = list()
        self.items = list()

    def __getitem__(self, item: int) -> Iterable[str]:
        if self.compressed:
            return self.sequence(item)
        else:
            return self.data[item]

    @classmethod
    def _check_savefiles(cls, name: str) -> List[str]:
        data_dir_contents = os.listdir(cls.savefile_loc)
        if not data_dir_contents:
            return None
        savefiles = [cls.savefile_loc + "/" + file for file in data_dir_contents if file.partition("-")[0] == name]
        if not savefiles:
            return None
        return sorted(savefiles, key=os.path.getmtime, reverse=True)
        
    def create(self, processor: Callable[[str, int], List], source: str, amount: Optional[int] = 0) -> None:
        # Ask the user if they want to load previous data if it exists
        savefiles = self._check_savefiles(self.name)
        if savefiles:
            reload = ""
            while reload.lower() not in ['y', 'n']:
                reload = input("There are dataset savefiles available, reload from earliest ? (y/n) : ")
            if reload == 'y':
                self.load(savefiles[0])
                return None
            else:
                print("Creating new sequence dataset...")
        self.data = processor(source, amount)

    @classmethod
    def load(cls, source: str = None) -> None:
        if not os.path.isfile(source):
            err = "Dataset savefile not found. (Supplied path : " + (f"'{source}'" if source else "(none)") + ")"
            raise ValueError(err)
        print("Loading save data from file :", source)
        res = cls("dummy")
        with open(source, 'r') as in_f:
            input = json.load(in_f)
            header = input["header"]
            res.name = header["name"]
            res.compressed = header["compression"]
            res.items = header["items"]
            res.data = input["data"]
        print("Savefile loaded.")
        return res

    @classmethod
    def load_latest(cls, name: str) -> None:
        savefiles = cls._check_savefiles(name)
        if not savefiles:
            print(f"No savefiles found for dataset : {name}")
            return
        return cls.load(savefiles[0])

    def save(self) -> None:
        filename = self.name + '-' + datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = self.savefile_loc + '/' + filename + ".json"
        print(f"Saving dataset to file : {save_path}")
        with open(save_path, 'w') as out_f:
            header = dict()
            header["name"] = self.name
            if not self.compressed:
                print("Warning : saving uncompressed data")
            header["compression"] = self.compressed
            header["items"] = self.items
            output = dict()
            output["header"] = header
            output["data"] = self.data
            json.dump(output, out_f)
    
    def bake_itemset(self) -> None:
        print(f"Computing set of sequence items for {self.name} ...")
        itemset = set()
        for row in self.data:
            for item in row:
                itemset.add(item)
        self.items = sorted(list(itemset))

    # Uses the sorted set of API calls to compress the data to mere integers
    def compress(self) -> List[str]:
        if not self.items:
            print("Compressing requires baking the dataset's itemset !")
            self.bake_itemset()
        print("Compressing data...")
        self.data =  [[self.items.index(item) for item in sequence] for sequence in self.data]
        self.compressed = True
        return self.items

    def _convert_sequence(self, seq: Sequence, other_items: Items_List) -> Iterable[int]:
        convert = lambda item: other_items.index(self.items[item])
        return [convert(item) for item in seq]

    def convert_calls(self, other_items: Items_List):
        self.data = [self._convert_sequence(sequence, other_items) for sequence in self.data]
        self.items = other_items

    def sequence(self, index: int, items: List[str] = None) -> Iterable[str]:
        if not self.compressed:
            raise TypeError("Can only decompress compressed data.")
        if not items:
            items = self.items
        return [items[item] for item in self.data[index]]

    # Million call long sequences that have very long stretches of similar data
    # aren't good for us, let's just kill them
    def remove_length_outliers(self) -> None:
        from numpy import percentile
        dlengths = map(len, self.data)
        q75, q25 = percentile(list(dlengths), [75, 25])
        iqr = q75-q25
        lmax = q75 + (1.5 * iqr)
        lmin = q25 - (1.5 * iqr)
        # Get rid of outliers by rebuilding the sequencer's data
        self.data = [seq for seq in self.data if len(seq) >= lmin and len(seq) <= lmax]

    # we want to be able to split the dataset two ways : 
    # training data / test data : ratio
    # data sections for each HMM, so we get interesting models : folds = n_models
    @classmethod
    def split(cls, i_range, ratio: float, folds: int, rng_f = None) -> Index_Split:
        train_max = floor(len(list(i_range)) * ratio)
        # get all possible indices and mix em up
        i: Total_Indices = list(i_range)
        shuffle(i, rng_f)

        train_i: List[int] = i[:train_max]
        test_i: Testing_Indices = i[train_max:]
        # make folds in training data
        train_i: Training_Indices = [a.tolist() for a in array_split(train_i, folds)]

        return (test_i, train_i)
