import sys

from transformers import AutoTokenizer


dataset = sys.argv[1]
model_name_or_path = sys.argv[2]
max_len = int(sys.argv[3])

subword_len_counter = 0

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
max_len -= tokenizer.num_special_tokens_to_add()

with open(dataset) as f_p:
    while True:
        try:
            line = next(f_p, None)
            if not line:
                break
        except Exception:
            pass
            
        line = line.rstrip()
        
        is_empty_line = (not line)
        print(line)
        subword_len_counter = 0
        
        if is_empty_line:
            continue
        
        token = line.split()[0]
        
        current_subwords_len = len(tokenizer.tokenize(token))
        
        # Token contains strange control characters like \x96 or \x95
        # Just filter out the complete line
        no_tokens_check = (current_subwords_len == 0)
        
        if no_tokens_check:
            continue
        
        should_overflow = (subword_len_counter + current_subwords_len) > max_len
        
        if should_overflow:
            print()
            print(line)
            subword_len_counter = current_subwords_len
            
            continue
        
        subword_len_counter += current_subwords_len
        
        print(line)