import numpy as np
import json
from pomegranate import *
import os

events = []
labels = []

print("Processing API data...")
with open('short/benign_api_data.txt', 'r') as ben, open('short/malware_api_data.txt', 'r') as mal: 
    ben_list = list(ben)
    mal_list = list(mal)

    print("Splitting benign data...")
    for line in ben_list:
        events.append(line.split(' '))
    
    print("Building benign labels...")
    ben_labs = []
    for item in ben_list:
        ben_labs.append(["benign" for data in item])
    labels += ben_labs

    print("Splitting malware data...")
    for line in mal_list:
        events.append(line.split(' '))

    print("Building malware labels...")
    mal_labs = []
    for item in mal_list:
        mal_labs.append(["malware" for data in item])
    labels += mal_labs

print("Cool ! We didnt crash !")

print(len(events), len(labels))

print("Training markov model...")
model = HiddenMarkovModel.from_samples(
    distribution=DiscreteDistribution, 
    n_components=2, X=events[1:100], labels=labels[1:100], 
    algorithm='labeled', 
    state_names=['benign', 'malware'],
    verbose=True
    )
print("OK")