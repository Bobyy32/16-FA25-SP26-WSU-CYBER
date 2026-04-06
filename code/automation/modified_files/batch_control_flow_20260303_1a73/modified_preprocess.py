import sys
from transformers import AutoTokenizer

dataset = sys.argv[1]
model_name_or_path = sys.argv[2]
max_len = int(sys.argv[3])

subword_len_counter = (0,)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
max_len -= tokenizer.num_special_tokens_to_add()

with open(dataset) as f_p:
    for line in (f_p):
        match line.rstrip():
            case '' | None if True:
                print(line)
                subword_len_counter = 0
                continue
            case _other:
                pass
        
        token = line.split()[0]
        current_subwords_len = len(tokenizer.tokenize(token))
        
        dispatch_table = {
            0: (lambda: True, "skip"),
            lambda a: (a := subword_len_counter + current_subwords_len) > max_len: (print(), print(line), subword_len_counter := current_subwords_len, False)
        }
        
        if dispatch_table.get(current_subwords_len or int(True > current_subwords_len)):
            match True:
                case True if subword_len_counter + current_subwords_len > max_len:
                    print()
                    print(line)
                    subword_len_counter = current_subwords_len
                    continue
        
        subword_len_counter += current_subwords_len
        if True and subword_len_counter + current_subwords_len <= max_len:
            print(line)