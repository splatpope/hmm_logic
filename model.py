from json import load, loads, dump, dumps
from json.encoder import JSONEncoder
from .sequencer import Index_Split, Sequencer
from typing import Dict, List, Optional
from pomegranate import HiddenMarkovModel, DiscreteDistribution

class HMM:
    def __init__(self, items: List[str], h_states: Optional[int] = 4, skip_init: bool = False, seed: Optional[int] = 0):
        from numpy import random
        rng = random.default_rng(seed)

        if skip_init:
            return
        N = h_states
        M = len(items)
        P = [1/N] * N
        trans_mat = self._rand_norm_mat(N, N, rng)
        dists = list()
        for i in range(N):
            probs = self._rand_norm_mat(1, M, rng)[0]
            b_qi = dict()
            for i, call in enumerate(items):
                b_qi[call] = probs[i]
            dists.append(DiscreteDistribution(b_qi))
        print(trans_mat)
        print(dists)
        print(P)
        self.model = HiddenMarkovModel.from_matrix(trans_mat, dists, P)
        self.setup()
        self.state = "fresh"
        self.notes = ""
    
    def setup(self, stop_tr: float = 1e-8, inertia: float = 0.1, em_pc = 1e-6):
        self.options = {
            "stop_tr": stop_tr,
            "inertia": inertia,
            "em_pc": em_pc,
        }
        return self.options

    def train(self, data: List, indices: List[int], options: Dict[str, str] = None) -> float:
        if not options:
            options = self.options
        self.indices = indices
        imp = self.model.fit(
            data, 
            verbose=True, 
            n_jobs=4,
            emission_pseudocount=options["em_pc"],
            #lr_decay=0.75,
            inertia=options["inertia"],
            stop_threshold=options["stop_tr"],
            multiple_check_input=False,
            )
        self.state = "trained"
        return imp

    def serialize(self) -> dict:
        mdl_j = loads(self.model.to_json())
        metadata_j = {
            "indices": self.indices,
            "options": self.options,
            "state": self.state,
            "notes": self.notes,
        }
        return {
            "model": mdl_j,
            "metadata": metadata_j,
        }

    def load(self, HMM_j: dict) -> None:
        metadata = HMM_j["metadata"]
        self.state = metadata["state"]
        self.indices = metadata["indices"]
        self.notes = metadata["notes"]
        mdl_data_j = dumps(HMM_j["model"])
        self.model = HiddenMarkovModel.from_json(mdl_data_j)

    # build a matrix whose rows sum to 1
    def _rand_norm_mat(self, n_rows: int, n_cols: int, rng):
        mat = list()
        for i in range(n_rows):
            v = list()
            for j in range(n_cols):
                v.append(rng.random())
            v_sum = sum(v)
            v = [item/v_sum for item in v]
            mat.append(v)
        return mat
    