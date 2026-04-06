import sys

from transformers import AutoTokenizer


dataset = sys.argv[1]
model_name_or_path = sys.argv[2]
max_len = int(sys.argv[3])

subword_len_counter = 0

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
max_len -= tokenizer.num_special_tokens_to_add()

with open(dataset) as f_p:
    for line in f_p:
        if not line.rstrip():
            print(line)
            subword_len_counter = 0
        else:
            token = line.split()[0]
            current_subwords_len = len(tokenizer.tokenize(token))
            if not current_subwords_len:
                continue
            elif (subword_len_counter + current_subwords_len) > max_len:
                print()
                print(line)
                subword_len_counter = current_subwords_len
            else:
                subword_len_counter += current_subwords_len
                print(line)


print(subword_len_counter)