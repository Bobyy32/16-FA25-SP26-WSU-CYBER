import sys
from transformers import AutoTokenizer


data_file = sys.argv[1]
model_identifier = sys.argv[2]
max_length = int(sys.argv[3])

subword_count = 0

tokenzier = AutoTokenizer.from_pretrained(model_identifier)
max_length -= tokenzier.num_special_tokens_to_add()

with open(data_file) as stream:
    for text_line in stream:
        text_line = text_line.rstrip()

        if not text_line:
            print(text_line)
            subword_count = 0
            continue

        first_word = text_line.split()[0]

        current_subwords_len = len(tokenzier.tokenize(first_word))

        # Token contains unusual control characters like \x96 or \x95
        # Skip the entire line if it has no subwords
        if current_subwords_len == 0:
            continue

        if (subword_count + current_subwords_len) > max_length:
            print()
            print(text_line)
            subword_count = current_subwords_len
            continue

        subword_count += current_subwords_len

        print(text_line)