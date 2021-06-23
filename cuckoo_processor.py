import json
import os
from typing import List, Optional

# Extract data from cuckoo reports

def process_reports(source: str, amount: Optional[int] = 0) -> List[str]:
    """Process cuckoo reports to get sequences of API calls.

    Args:
        source (str): Path to the report(s).
        amount (Optional[int], optional): Quantity of reports to process. Defaults to 0 (all of them).

    Returns:
        List[str]: The API call sequences extracted from the reports.
    """
    # Build a list of files if we were handed a folder path
    if not os.path.isdir(source):
            folder = ""
            source = [source]
    else:
        folder = source
        if folder[-1] != "/":
            folder += "/"
        source = [report_f for report_f in os.listdir(folder)]
    counter = 0
    reports_quantity = len(source)
    data = list()
    for filename in source:
        print(f"Processing file : {filename}, ~~~ ({str(counter + 1)}/{str(reports_quantity)})")
        # Load report file and process it
        counter += 1
        with open(folder + filename, "r") as in_f:
            report = dict()
            try:
                report = json.load(in_f)
            except:
                print("Couldn't load report, ignoring...")
                continue
            if 'behavior' not in report:  # Interesting data is in behavior.processes.calls.api
                continue
            processes_data = report['behavior']['processes']
            # Iterate over all processes of the report.
            for process in processes_data:
                api_calls = process['calls']
                if api_calls:
                    call_data = [call['api'].lower() for call in api_calls]
                    data.append(call_data)
        if amount == 0:
            continue
        if counter >= amount:
            break
    print(f"Processed {counter} reports.")
    return data