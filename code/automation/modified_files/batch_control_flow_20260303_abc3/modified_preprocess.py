import sys

from transformers import AutoTokenizer


DatasetFile = sys.argv[1]
ModelPath = sys.argv[2]
MaxLength = int(sys.argv[3])

SubwordCounter = 0

tokenizer = AutoTokenizer.from_pretrained(ModelPath)
MaxLength -= tokenizer.num_special_tokens_to_add()


with open(DatasetFile) as f_p:
    for line in f_p:
        line = line.rstrip()

        if not line:
            print(line)
            SubwordCounter = 0
            continue

        token = line.split()[0]

        current_subwords_len = len(tokenizer.tokenize(token))

        # Filter lines with strange control characters like \x96 or \x95
        # Just skip the complete line here
        if current_subwords_len == 0:
            continue

        if SubwordCounter + current_subwords_len > MaxLength:
            print()
            print(line)
            SubwordCounter = current_subwords_len
            continue

        SubwordCounter += current_subwords_len

        print(line)