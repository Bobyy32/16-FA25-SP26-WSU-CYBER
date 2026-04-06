import sys
from transformers import AutoTokenizer


dataset = sys.argv[1]
model_name_or_path = sys.argv[2]
max_len = int(sys.argv[3])

subword_len_counter = 0

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
max_len -= tokenizer.num_special_tokens_to_add()

with open(dataset) as f_p:
    for line in f_p.rstrip():
        if not line:
            print(line)
            subword_len_counter = 0
            continue

        token, current_subwords_len = (line.split()[0], len(tokenizer.tokenize(line.split()[0])))
        is_empty_or_overflow = current_subwords_len == 0 or (subword_len_counter + current_subwords_len) > max_len

        if is_empty_or_overflow:
            print() if subword_len_counter := current_subwords_len else None
            continue

        subword_len_counter += current_subwords_len
        print(line)