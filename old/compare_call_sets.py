import json

with open('call_sets/mal.json', 'r') as mal, open('call_sets/ben.json', 'r') as ben:
    good = set(json.load(ben))
    bad = set(json.load(mal))
    print("\nGOOD CALLS", good)
    print("\nBAD CALLS", bad)
    print("\nGOOD AND BAD", good.intersection(bad))
    print("\nONLY GOOD", good.difference(bad))
    print("\nONLY BAD", bad.difference(good))