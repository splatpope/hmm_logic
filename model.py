from json import loads, dumps
from typing import Any, Dict, List, Optional

from numpy.random import default_rng
from pomegranate import HiddenMarkovModel, DiscreteDistribution

# redefine this to suit your computer
MACHINE_THREADS = 4

class HMM:
    def __init__(self, items: List[str], h_states: Optional[int] = 4, B = None, rng = None):
        if not rng:
            rng = default_rng()
        self.rng = rng
        o_states = len(items)
        P = [1/h_states] * h_states  # init hidden probs generated uniformly
        trans_mat = self._rand_norm_mat(h_states, h_states)
        dists = list()
        if not B:  # generate emissions randomly
            for i in range(h_states):
                probs = self._rand_norm_mat(1, o_states)[0]
                b_qi = dict()
                for index, call in enumerate(items):
                    b_qi[call] = probs[index]
                dists.append(DiscreteDistribution(b_qi))
        else:
            dists = [DiscreteDistribution(b_qi) for b_qi in B]
        print("Transition matrix:")
        print(trans_mat)
        self.model = HiddenMarkovModel.from_matrix(trans_mat, dists, P)
        self.setup_default_options()
        self.state = "fresh"
        self.notes = ""

    def setup_default_options(self, stop_tr: float = 1e-6, inertia: float = 1e-3, em_pc: float = 1e-8):
        self.options = {
            "stop_tr": stop_tr,
            "inertia": inertia,
            "em_pc": em_pc,
            "verbose": True,
        }
        return self.options

    def train(self, data: List, options: Dict[str, Any] = None) -> float:
        self.options.update(options)
        imp = self.model.fit(data,
            verbose=self.options["verbose"],
            n_jobs=MACHINE_THREADS,
            emission_pseudocount=self.options["em_pc"],
            #lr_decay=0.75,
            inertia=self.options["inertia"],
            stop_threshold=self.options["stop_tr"],
            #multiple_check_input=False,
            )
        self.state = "trained"
        return imp

    def serialize(self) -> dict:
        mdl_j = loads(self.model.to_json())
        metadata_j = {
            "options": self.options,
            "state": self.state,
            "notes": self.notes,
        }
        return {
            "model": mdl_j,
            "metadata": metadata_j,
        }

    @classmethod
    def load(cls, HMM_j: dict, rng = None) -> None:
        mdl = cls.__new__(cls)
        if not rng:
            rng = default_rng()
        mdl.rng = rng
        metadata = HMM_j["metadata"]
        mdl.options = metadata["options"]
        mdl.state = metadata["state"]
        mdl.notes = metadata["notes"]
        mdl_data_j = dumps(HMM_j["model"])
        mdl.model = HiddenMarkovModel.from_json(mdl_data_j)
        if not mdl.options:
            mdl.setup_default_options()
        return mdl

    # build a matrix whose rows sum to 1
    def _rand_norm_mat(self, n_rows: int, n_cols: int):
        mat = list()
        for i in range(n_rows):
            row = list()
            for j in range(n_cols):
                row.append(self.rng.random())
            row_sum = sum(row)
            row = [item/row_sum for item in row]
            mat.append(row)
        return mat
    