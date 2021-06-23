with open("long/benign_api_data.txt", "r") as ben_long, open("long/malware_api_data.txt", "r") as mal_long, open("short/benign_api_data.txt", "w") as ben_short, open("short/malware_api_data.txt", "w") as mal_short:
    counter_ben = 0
    counter_mal = 0
    for line in ben_long:
        if len(line) < 100000:
            ben_short.write(line)
        else:
            counter_ben += 1
            print("Chopped off long benign line")

    for line in mal_long:
        if len(line) < 100000:
            mal_short.write(line)
        else:
            counter_mal += 1
            print("Chopped off long malware line")

    print("Chopped off", counter_ben, "benign lines")
    print("Chopped off", counter_mal, "malware lines")
