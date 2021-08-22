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
    models_path = "./data/model_banks/"

    def __init__(self, datasets: List[Sequencer] = None) -> None:

        self.seed = time_ns()
        self.datasets = datasets
        if not self.datasets:
            print("Please try again with datasets this time.")
            raise ValueError()
        itemset = set()
        for dataset in self.datasets:
            itemset = itemset.union(set(dataset.items))
        self.items = sorted(list(itemset))
        self.rng = default_rng(self.seed)
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

    def setup(self, dataset: Sequencer, n_models: int = 6, h_states = 4, indices: Index_Split = None) -> Index_Split:
        # try to guess emission probabilities from data
        if self.state == "bad data":
            print("Data not in proper format, please call _generalize_datasets.")
            return None
        if not indices:  # default index split : 1 segment by model
            indices = Sequencer.split(range(len(dataset.data)), 1.0, n_models)
        t_indices = indices[1]
        self.models = list()
        # make n_states segments from mdl_indices and calculate distribution of items for initialisation of B
        for mdl_indices in t_indices:
            emission_matrix = list()
            segments = Sequencer.split(mdl_indices, 1.0, h_states)[1]
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
        return indices


    def train_models(self, dataset: Sequencer, indices: Index_Split, options = None) -> None:
        if not self.state == "ready":
            print("Detector not ready, please call setup.")
            return None
        print("Training models...")
        for mdl, index_list in zip(self.models, indices[1]):
            mdl.train(data=[dataset.sequence(i) for i in index_list],
                indices=index_list,
                options=options,
                )

    def save(self, name, indices):
        with open(self.models_path + name + ".json", "w") as out_f:
            out_j = {
                "models": [mdl.serialize() for mdl in self.models],
                "notes": self.notes,
                "items": self.items,
                "datasets": [ds.name for ds in self.datasets],
                "seed": self.seed,
            }
            json.dump(out_j, out_f)

    @classmethod
    def load(cls, name):
        dec = cls.__new__(cls)
        with open(cls.models_path + name + ".json", "r") as in_f:
            try:
                in_j = json.load(in_f)
                dec.items = in_j["items"]
                dec.notes = in_j["notes"]
                dec.seed = in_j["seed"]
                self.rng = default_rng(dec.seed)
                dec.datasets = [Sequencer.load_latest(name) for name in in_j["datasets"]]
                dec.models = [HMM.load(hmm_j, self.rng) for hmm_j in in_j["models"]]
            except KeyError as error:
                print("Detector save file incompatible :", error)
        return dec

    def create_score_dataset(self) -> Scores:
        scores = dict()
        print("Computing probability scores...")
        for dataset in self.datasets:
            ds_scores = list()
            for i in range(len(dataset.data)):
                seq_score = [str(hmm.model.log_probability(dataset.sequence(i))) for hmm in self.models]
                ds_scores.append(seq_score)
            scores[dataset.name] = ds_scores

        return scores

    def _make_arff_header(self) -> str:
        out = "@RELATION hmm_score\n\n"
        for i in range(len(self.models)):
            out += "@ATTRIBUTE hmm_" + str(i) + "\tNUMERIC\n"
        out += "@ATTRIBUTE class\t{"
        classes = ','.join([ds.name for ds in self.datasets])
        out += classes
        out += "}\n"
        return out

    def write_arff(self, scores: Scores) -> None:
        filename = "scores" + '-' + datetime.now().strftime("%Y%m%d-%H%M%S") + ".arff"

        sep = ','
        print("Writing scores in " + filename + "...")
        with open("./data/hmm_scores/" + filename, 'w') as out_f:
            out_f.write(self._make_arff_header())
            out_f.write("\n@DATA\n")
            # write scores for all datasets
            for ds_name, ds_scores in scores.items():
                line = ''
                for seq_score in ds_scores:
                    line += sep.join(seq_score) + sep + ds_name + '\n'
                out_f.write(line)
