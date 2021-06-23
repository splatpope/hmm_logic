import json
import argparse
import os

parser = argparse.ArgumentParser(description="Create an api call sequence dataset from json cuckoo reports")

parser.add_argument('filenames', nargs='+', help="Name of json cuckoo report (can use globs)")
parser.add_argument('-n', '--number', default=-1, type=int, help="Number of reports to process")
parser.add_argument('-r', '--resume', default=-1, type=int, help="Resume at nth report")

args = parser.parse_args()


counter = 0
with open('long/malware_api_data.txt', 'w') as out:
    for filename in args.filenames:
        counter += 1
        if counter < args.resume:
            continue
        print(filename)

        with open(filename, 'r') as f:
            # Load report file
            try:
                report = json.load(f)
            except:
                continue
            # Extract process behavior data
            if 'behavior' not in report:
                continue
            processes_data = report['behavior']['processes']

            # Some programs call multiple process, but not all of them have api call data
            for process in processes_data:
                # Extract api call data from the process
                api_calls = process['calls']
                if api_calls: # may be empty
                    call_data = [call['api'].lower() for call in api_calls]
                    out.write(' '.join(call_data) + '\n')
                    out.flush()
                    os.fsync(out)

        if args.number == -1:
            continue
        if counter >= args.number:
            break

print("Processed", counter, "reports")