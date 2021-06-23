import json
print("===== CALL USAGE =====")
calls_mal = set()
calls_ben = set()

print("Loading callsets...")
with open("call_sets/ben.json", "r") as call_ben_f, open("call_sets/mal.json", "r") as call_mal_f:
    callset_ben = set(json.load(call_ben_f))
    callset_mal = set(json.load(call_mal_f))

    callset = callset_ben.union(callset_mal)
    calls_ben = dict.fromkeys(callset, 0)
    calls_mal = dict.fromkeys(callset, 0)

print("Computing call statistics for benign apps...")
with open("long/benign_api_data.txt", "r") as ben_data:
    for line in ben_data:
        for call in line.rstrip().split(' '):
            if call in calls_ben:
                calls_ben[call] += 1

print("Computing call statistics for malware apps...")
with open("long/malware_api_data.txt", "r") as mal_data:
    for line in mal_data:
        for call in line.rstrip().split(' '):
            if call in calls_mal:
                calls_mal[call] += 1


print("Saving call stats...")
stats = dict()
stats["ben"] = calls_ben
stats["mal"] = calls_mal
with open("stats/api_call.json", "w") as call_stats_f:
    json.dump(stats, call_stats_f, indent=4)
