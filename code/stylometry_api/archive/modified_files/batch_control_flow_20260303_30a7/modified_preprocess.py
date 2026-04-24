import sys

from transformers import AutoTokenizer


dataset = sys.argv[1]
model_name_or_path = sys.argv[2]
max_len = int(sys.argv[3])

subword_len_counter = 0

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
max_len -= tokenizer.num_special_tokens_to_add()

with open(dataset, mode='r') as f_p:
    read_lines = [line.rstrip() for line in f_p.readlines()]
    
    for i, line in enumerate(read_lines):
        if not line:
            print(line)
            subword_len_counter = 0
            continue
        else:
            token = line.split()[0]
            
            current_subwords_len = len(tokenizer.tokenize(token))
            
            if current_subwords_len == 0:
                continue
            
            elif (subword_len_counter + current_subwords_len) > max_len:
                print()
                print(line)
                subword_len_counter = current_subwords_len
                continue
            
            else:
                subword_len_counter += current_subwords_len
                print(line)