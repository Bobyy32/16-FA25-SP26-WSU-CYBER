import sys

from transformers import AutoTokenizer


dataset_file = sys.argv[1]
model_name_or_path = sys.argv[2]
maxLength = int(sys.argv[3])

subwordLenCounter = 0

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
maxLength -= tokenizer.num_special_tokens_to_add()

with open(dataset_file) as fileHandle:
    for line in fileHandle:
        line = line.rstrip()

        if not line:
            print(line)
            subwordLenCounter = 0
            continue

        token = line.split()[0]

        currentSubWordsLength = len(tokenizer.tokenize(token))

        # Exclude lines with odd control characters such as \x96 or \x95
        if currentSubWordsLength == 0:
            continue

        if (subwordLenCounter + currentSubWordsLength) > maxLength:
            print()
            print(line)
            subwordLenCounter = currentSubWordsLength
            continue

        subwordLenCounter += currentSubWordsLength

        print(line)