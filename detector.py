import json
from typing import Dict, List
from datetime import datetime

from numpy.random import default_rng
from time import time_ns

from .sequencer import Index_Split, Sequencer
from .model import HMM

HMMScores = List[int]
Scores = Dict[str, List[HMMScores]]

class Detector:
    savefile_loc = "./data/detectors/"

    def __init__(self, *datasets: List[Sequencer]) -> None:

        self.seed = time_ns()
        self.rng = default_rng(self.seed)
        self.datasets = datasets
        if not self.datasets:
            print("Please try again with datasets this time.")
            raise ValueError()
        itemset = set()
        for dataset in self.datasets:
            itemset = itemset.union(set(dataset.items))
        self.items = sorted(list(itemset))
        self.notes = ""
        self.state = "fresh"

        if self._check_for_dataset_generality():
            print("All good. Please call setup to initialize models.")
        else:
            self.state = "bad data"
            print("The datasets have different item sets ! Please call _generalize_datasets, then setup.")

    def _check_for_dataset_generality(self) -> bool:
        itemsets = [set(ds.items) for ds in self.datasets]
        total_itemset = set(self.items)
        for itemset in itemsets:
            if len(total_itemset.difference(itemset)):
                return False
        return True

    def _generalize_datasets(self) -> None:
        if not self.state == "bad data":
            print("Nothing to do here.")
        for dataset in self.datasets:
            dataset.convert_calls(self.items)
        self.state = "fresh"
    
    def _prevent_states(self, *states) -> bool:
        if self.state in states:
            bad_states = {
                'fresh': 'Detector not ready, please call setup.',
                'bad data': 'Data not in proper format, please call _generalize_datasets.',
                'ready' : 'Somehow, being ready is bad here. Go figure.'
            }
            print(bad_states[self.state])
            return True
        else:
            return False

    def setup(self, dataset: Sequencer, 
            n_models: int = 6, 
            h_states = 4, 
            indices: Index_Split = None,
            randomB = False, 
        ) -> None:
        # instanciate hmms
        if self._prevent_states("bad_data"):
            return None
        if not indices:  # default index split : 1 segment by model, no testing
            print("Using default index split : 70% training, 30% testing")
            indices = Sequencer.split(range(len(dataset.data)), 0.7, n_models, self.rng.random)
        train_indices = indices[1]
        print("Initializing models...")
        self.models = list()
        for mdl_indices in train_indices:
            if not randomB:
                # make n_states segments from mdl_indices and calculate distribution of items for initialisation of B
                emission_matrix = list()
                segments = Sequencer.split(mdl_indices, 1.0, h_states, self.rng.random)[1]
                for segment in segments:
                    seg_data = [dataset.data[i] for i in segment]  # gather segment data
                    seg_item_count = [0 for i in range(len(self.items))]  # inital item count: 0 for all
                    for seg_seq in seg_data:  # count items in each sequence
                        for seg_data_item in seg_seq:
                            seg_item_count[seg_data_item] += 1
                    seg_total_count = sum(seg_item_count)  # normalize
                    seg_item_probs = [item_count/seg_total_count for item_count in seg_item_count]
                    emissions = dict()
                    for i, item in enumerate(self.items):
                        emissions[item] = seg_item_probs[i]
                    emission_matrix.append(emissions)
            self.models.append(HMM(self.items, h_states, emission_matrix, self.rng))
        self.state = "ready"
        self.indices = indices
        print("Detector ready for training.")

    def train_models(self, dataset: Sequencer, indices: Index_Split = None, options = None) -> None:
        if self._prevent_states("fresh", "bad_data"):
            return None
        if not indices:
            indices = self.indices
        else:
            print("Overriding index split for dataset "+dataset.name +"...")
            self.indices = indices
        print("Training models...")
        for mdl, index_list in zip(self.models, indices[1]):
            mdl.train(data=[dataset.sequence(i) for i in index_list], options=options)

    def save(self, name):
        with open(self.savefile_loc + name + ".json", "w") as out_f:
            out_j = {
                "datasets": [ds.name for ds in self.datasets],
                "state": self.state,
                "notes": self.notes,
                "seed": self.seed,
                "items": self.items,
                "indices": self.indices,
                "models": [mdl.serialize() for mdl in self.models],
            }
            json.dump(out_j, out_f)

    @classmethod
    def load(cls, name):
        dec = cls.__new__(cls)
        with open(cls.savefile_loc + name + ".json", "r") as in_f:
            try:
                in_j = json.load(in_f)
                dec.datasets = [Sequencer.load_latest(name) for name in in_j["datasets"]]
                dec.state = in_j["state"]
                dec.notes = in_j["notes"]
                dec.seed = in_j["seed"]
                dec.rng = default_rng(dec.seed)
                dec.items = in_j["items"]
                dec.indices = in_j["indices"]
                dec.models = [HMM.load(hmm_j, dec.rng) for hmm_j in in_j["models"]]
            except KeyError as error:
                print("Detector save file incompatible :", error)
        return dec

    def create_score_dataset(self, dataset, indices, adjusted = False) -> list:
        if self._prevent_states("fresh", "bad_data"):
            return False
        print("Computing probability scores...")
        ds_scores = list()
        for i in indices: 
            seq = dataset.sequence(i)
            factor = 1.0 if not adjusted else 1.0/len(seq)
            seq_score = [factor * hmm.model.log_probability(seq) for hmm in self.models]
            seq_score += [dataset.name]
            ds_scores.append(seq_score)
        return ds_scores

    def _make_arff_header(self, name) -> str:
        out = "@RELATION hmm_score_" + name + "\n\n"
        for i in range(len(self.models)):
            out += "@ATTRIBUTE hmm_" + str(i) + "\tNUMERIC\n"
        out += "@ATTRIBUTE class\t{"
        classes = ','.join([ds.name for ds in self.datasets])
        out += classes
        out += "}\n"
        return out

    def write_arff(self, name, scores: Scores) -> None:
        filename = "scores" + '-' + name + '-' + datetime.now().strftime("%Y%m%d-%H%M%S") + ".arff"
        sep = ','
        print("Writing scores in " + filename + "...")
        with open("./data/hmm_scores/" + filename, 'w') as out_f:
            out_f.write(self._make_arff_header(name))
            out_f.write("\n@DATA\n")
            # write scores for all datasets
            for seq_score in scores:
                score_str = [str(feat) for feat in seq_score]                
                line = sep.join(score_str) + '\n'
                out_f.write(line)
