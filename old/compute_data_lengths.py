print("===== COMPUTE DATA LENGTHS =====")
print("Processing API data...")
with open("long/benign_api_data.txt", "r") as ben_long, open("long/malware_api_data.txt", "r") as mal_long:
    print("Converting benign data to list...")
    ben_list_long = list(ben_long)
    print("Converting malware data to list...")
    mal_list_long = list(mal_long)

    print("Computing benign data lengths...")
    ben_lens_long = [len(item.rstrip().split(' ')) for item in ben_list_long]
    print("Computing malware data lengths...")
    mal_lens_long = [len(item.rstrip().split(' ')) for item in mal_list_long]


    print("Writing data lengths...")
    with open("long/benign_data_lengths.txt", "w") as ben_lens_long_f, open("long/malware_data_lengths.txt", "w") as mal_lens_long_f:
        for length in ben_lens_long:
            ben_lens_long_f.write(str(length) + "\n")
        for length in mal_lens_long:
            mal_lens_long_f.write(str(length) + "\n")
print("OK")

