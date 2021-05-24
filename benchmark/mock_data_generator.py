# Generate set of mock text data.

import os

dirname = "data/"
mock_text = open('mock_data.txt', "r").read()

os.makedirs(os.path.dirname(dirname), exist_ok=True)

for i in range(20):
    # Ubuntu 16.04 python3.5 not supports f function
    with open(dirname + "file_" + str(i) + ".txt", 'w') as f:
        f.write(mock_text)
