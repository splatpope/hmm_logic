import json
from json.encoder import JSONEncoder
from math import floor
from .sequencer import Index_Split, Sequencer
from .model import HMM

from typing import Dict, List

HMM_Scores = List[int]
Scores = Dict[str, List[HMM_Scores]]

class Detector:
    models_path = "./data/model_banks/"

    def __init__(self, n_models: int, datasets: List[Sequencer] = None, load_name: str = None) -> None:
        if load_name:
            self.load(load_name)
            return
        self.datasets = datasets
        if not self.datasets:
            print("Please try again with datasets this time")
            raise ValueError()
        itemset = set()
        for ds in self.datasets:
            itemset = itemset.union(set(ds.items)) 
        self.items = sorted(list(itemset))

        self.models = [HMM(self.items) for i in range(n_models)]
        self.notes = ""

        if self._check_for_dataset_generality():
            print("All good. Proceed to setup or train with default parameters.")

    def _check_for_dataset_generality(self) -> bool:
        itemsets = [set(ds.items) for ds in self.datasets]
        total_itemset = set(self.items)
        for itemset in itemsets:
            if itemset.difference(total_itemset):
                return False
        return True
        
    def _generalize_datasets(self) -> None:
        for ds in self.datasets:
            ds.convert_calls(self.items)

    def train_models(self, dataset: Sequencer, indices: Index_Split, stop_tr=1e-6) -> None:
        print("Training models...")
        for mdl, index_list in zip(self.models, indices[1]):
            mdl.train([dataset.sequence(i) for i in index_list], index_list)

    def save(self, name, indices):
        with open(self.models_path + name + ".json", "w") as out_f:
            out_j = {
                "models": [mdl.serialize() for mdl in self.models],
                "notes": self.notes,
                "items": self.items,
                "datasets": [ds.name for ds in self.datasets],
            }
            json.dump(out_j, out_f)

    def load(self, name):
        with open(self.models_path + name + ".json", "r") as in_f:
            in_j = json.load(in_f)
            self.items = in_j["items"]
            self.notes = in_j["notes"]
            self.datasets = [Sequencer(name).load_latest() for name in in_j["datasets"]]
            self.models = [HMM(["dummy"], skip_init=True).load(hmm_j) for hmm_j in in_j["models"]]
            
    def create_score_dataset(self) -> Scores:
        scores = dict()
        print("Computing probability scores...")
        for ds in self.datasets:
            ds_scores = list()
            for i in range(len(ds.data)):
                seq_score = [str(mdl.log_probability(ds.sequence(i))) for mdl in self.models]
                ds_scores.append(seq_score)
            scores[ds.name] = ds_scores

        return scores

    def write_arff(self, scores: Scores) -> None:
        from datetime import datetime
        filename = "scores" + '-' + datetime.now().strftime("%Y%m%d-%H%M%S") + ".arff"

        sep = ','
        print("Writing scores in " + filename + "...")
        with open("./data/hmm_scores/" + filename, 'w') as out_f:
            out_f.write("@RELATION hmm_score\n\n")
            # determine features based on number of models
            for i in range(len(self.models)):
                out_f.write("@ATTRIBUTE hmm_" + str(i) + "\tNUMERIC\n")
            out_f.write("@ATTRIBUTE class\t{")
            classes = sep.join([ds.name for ds in self.datasets])
            out_f.write("}\n")

            out_f.write("\n@DATA\n")
            # write scores for all datasets
            for ds_name, ds_scores in scores.items():
                line = ''
                for seq_score in ds_scores:
                    line += sep.join(seq_score) + sep + ds_name + '\n'
                out_f.write(line)
