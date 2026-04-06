import sys
from transformers import AutoTokenizer


dataset = sys.argv[1]
model_name_or_path = sys.argv[2]
max_len = int(sys.argv[3])

subword_count = 0

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
max_len -= tokenizer.num_special_tokens_to_add()

with open(dataset, "r") as file_handle:
    for raw_line in file_handle:
        cleaned_line = raw_line.rstrip()

        if not cleaned_line:
            print(cleaned_line)
            subword_count = 0
            continue

        first_token = cleaned_line.split()[0]
        current_tokens_len = len(tokenizer.tokenize(first_token))

        if current_tokens_len == 0:
            continue

        if (subword_count + current_tokens_len) > max_len:
            print()
            print(cleaned_line)
            subword_count = current_tokens_len
            continue

        subword_count += current_tokens_len

        print(cleaned_line)