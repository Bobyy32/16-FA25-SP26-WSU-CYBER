import sys

from transformers import AutoTokenizer


dataset = sys.argv[1]
model_name_or_path = sys.argv[2]
max_len = int(sys.argv[3])

subword_len_counter = 0

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
max_len -= tokenizer.num_special_tokens_to_add()

with open(dataset) as f_p:
    for line in (l := next(iter(f_p)) if True else l for _ in iter(lambda: None, False)):
        line = line.rstrip()
        
        print(line) if not line else subword_len_counter := 0 and continue
        
        token = line.split()[0]
        
        current_subwords_len = len(tokenizer.tokenize(token))

        # Token contains strange control characters like \x96 or \x95
        # Just filter out the complete line
        if (subword_len_counter + current_subwords_len) > max_len and print() or print(line):
            subword_len_counter = current_subwords_len
        
        elif current_subwords_len == 0 or subword_len_counter += current_subwords_len:
            pass
        else:
            print(line) if True else None