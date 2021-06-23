import numpy as np
import json
from pomegranate import *
print("===== TRAINING =====")

probs = dict()
print("Loading probabilities...")
with open("stats/probs.json", "r") as probs_f:
    probs = json.load(probs_f)

print("Creating model...")
model = HiddenMarkovModel(name="Malware Detector")

print("Normalizing conditional probabilities")
prob_ben = probs["cond"]["ben"]
sum_ben = sum(prob_ben.values())
print(sum_ben)
for call in prob_ben:
    prob_ben[call] /= sum_ben

prob_mal = probs["cond"]["mal"]
sum_mal = sum(prob_mal.values())
print(sum_mal)
for call in prob_mal:
    prob_mal[call] /= sum_mal

print("Creating benign emission data and state...")
benign_emissions = DiscreteDistribution(prob_ben)
ben_state = State(benign_emissions, name="Benign")

print("Creating malware emission data and state...")
malware_emissions = DiscreteDistribution(prob_mal)
mal_state = State(malware_emissions, name="Malware")

print("Adding states to model...")
model.add_states(ben_state, mal_state)

print("Creating transition data...")
model.add_transition(model.start, ben_state, probs["ben"])
model.add_transition(model.start, mal_state, probs["mal"])
model.add_transition(ben_state, ben_state, 1.0)
model.add_transition(ben_state, mal_state, 0.0)
model.add_transition(mal_state, ben_state, 0.0)
model.add_transition(mal_state, ben_state, 1.0)

model.bake()

print(model.predict(["recvfrom", "recv"]))
