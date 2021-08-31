import os
from typing import Optional, List

def process_binaries(source: str, amount: Optional[int] = 0) -> List[str]:
    # for now, only process one binary
    chunk_size = 500

    if not os.path.isdir(source):
            folder = ""
            source = [source]
    else:
        folder = source
        if folder[-1] != "/":
            folder += "/"
        source = [report_f for report_f in os.listdir(folder)]

    bin_quantity = len(source)
    counter = 0
    data = list()

    for filename in source:
        print(f"Processing file : {filename}, ~~~ ({str(counter + 1)}/{str(bin_quantity)})")
        counter += 1
        with open(folder + filename, 'rb') as bin_f:
            byte_list = list(bin_f.read())

        data.append([str(item) for item in byte_list])

        if amount == 0:
            continue
        if counter >= amount:
            break

    return data

