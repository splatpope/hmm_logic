import json
print("===== GET API CALL LIST =====")
call_set_ben = set()
call_set_mal = set()

print("Processing API data...")
with open("long/benign_api_data.txt") as ben_f, open("long/malware_api_data.txt", "r") as mal_f:
    print("Collecting benign API calls...")
    for line in ben_f:
        for call in line.rstrip().split(' '):
            call_set_ben.add(call)

    print("Collecting malware API calls...")
    for line in mal_f:
        for call in line.rstrip().split(' '):
            call_set_mal.add(call)

print("Saving call sets...")
with open("call_sets/ben.json", "w") as ben_set_f, open("call_sets/mal.json", "w") as mal_set_f:
    json.dump(list(call_set_ben), ben_set_f, indent=4)
    json.dump(list(call_set_mal), mal_set_f, indent=4)