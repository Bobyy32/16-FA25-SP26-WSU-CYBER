from transformers import AutoTokenizer

import sys


payload = sys.argv[1]
model_identifier = sys.argv[2]
length_limit = int(sys.argv[3])

subword_length_tally = 0

token_processor = AutoTokenizer.from_pretrained(model_identifier)
length_limit -= token_processor.num_special_tokens_to_add()

with open(payload) as file_handler:
    for line in file_handler:
        line = line.rstrip().strip()

        if not line:
            print(line)
            subword_length_tally = 0
            continue

        token = line.split()[0]

        current_subwords_len = len(token_processor.tokenize(token))

        # Discard lines containing unusual control characters such as \x96 or \x95
        # Eliminate entire line for filtering purposes
        if current_subwords_len == 0:
            continue

        if (subword_length_tally + current_subwords_len) > length_limit:
            print()
            print(line)
            subword_length_tally = current_subwords_len
            continue

        subword_length_tally += current_subwords_len

        print(line)