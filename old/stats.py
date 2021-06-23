import json
print("===== STATS COMPENDIUM =====")
stats_usage = dict()
print("Loading api call usage stats...")
with open("stats/api_call.json", "r") as call_usage_f:
    stats_usage = json.load(call_usage_f)

call_set = list(stats_usage["ben"])
stats_probs = dict.fromkeys(stats_usage["ben"], 0)

ben_call_num = sum(stats_usage["ben"].values()) 
mal_call_num = sum(stats_usage["mal"].values())
total_calls = ben_call_num + mal_call_num
print("Total number of calls :", total_calls, "( Benign :", ben_call_num, ")", " ( Malware :", mal_call_num, ")")

ben_apps_num = 2275
mal_apps_num = 10911
apps_num = ben_apps_num + mal_apps_num
print("Total number of apps : ", apps_num, "( Benign :", ben_apps_num, ")", " ( Malware :", mal_apps_num, ")")

ben_lens = 0
mal_lens = 0
with open("long/benign_data_lengths.txt", "r") as ben_lens_f, open("long/malware_data_lengths.txt", "r") as mal_lens_f:
    ben_len_list = list(ben_lens_f)
    ben_lens = sum(int(item) for item in ben_len_list)
    mal_len_list = list(mal_lens_f)
    mal_lens = sum(int(item) for item in mal_len_list)
total_lens = ben_lens + mal_lens
print("Total sequence lengths :", total_lens, "( Benign :", ben_lens, ")", "( Malware :", mal_lens, ")")
print("OK" if total_lens == total_calls and ben_call_num == ben_lens and mal_call_num == mal_lens else "WRONG")


print("\nPROBABILITIES -----\n")

print("\nHidden states :")
prob_ben = ben_lens / total_lens
prob_mal = mal_lens / total_lens
print("P(qt = ben) =", prob_ben)
print("P(qt = mal) =", prob_mal)
temp = prob_ben + prob_mal
print("Sum :", temp, "///", "OK" if 1.0 - temp < 0.0000001 else "WRONG")

print("\nObserved symbols :")
for call in call_set:
    stats_probs[call] = (stats_usage["ben"][call] + stats_usage["mal"][call]) / total_calls # (n_k_b + n_k_m) / (n_b + n_m)
    print("P(ot = " + call + ") =", stats_probs[call])
temp = sum(stats_probs.values())
print("Sum :", temp, "///", "OK" if 1.0 - temp < 0.0000001 else "WRONG")

stats_probs_cond = dict()
stats_probs_cond["ben"] = dict()
stats_probs_cond["mal"] = dict()
print("\nConditional observations :")
for call in call_set:
    stats_probs_cond["ben"][call] = stats_usage["ben"][call] / ben_call_num # n_b / (n_b + n_m)
    stats_probs_cond["mal"][call] = stats_usage["mal"][call] / mal_call_num # n_m / (n_b + n_m)
    print("P(ot = " + call + " | qt = ben) =", stats_probs_cond["ben"][call], "~~~ P(ot = " + call + " | qt = mal) =", stats_probs_cond["mal"][call])

print("\nSaving calculated probabilities...")
with open("stats/probs.json", "w") as probs_f:
    probs = dict()
    probs["ben"] = prob_ben
    probs["mal"] = prob_mal
    probs["cond"] = {
        "ben" : stats_probs_cond["ben"], 
        "mal" : stats_probs_cond["mal"]
        }
    json.dump(probs, probs_f, indent=4)