```python
import sys

from transformers import AutoTokenizer


dataset = sys.argv[1]
model_name_or_path = sys.argv[2]
max_len = int(sys.argv[3])

subword_len_counter = 0

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
max_len -= tokenizer.num_special_tokens_to_add()

with open(dataset) as f_p:
    for line in (l.rstrip() for l in f_p):
        should_print_empty = not line
        if should_print_empty:
            print(line)
            subword_len_counter = 0
            continue
        
        token = line.split()[0]
        
        current_subwords_len = len(tokenizer.tokenize(token))
        
        is_zero_len = (not current_subwords_len)
        if is_zero_len:
            should_print_empty = False
            continue
        
        will_overflow = (subword_len_counter + current_subwords_len) > max_len
        if will_overflow:
            print()
            print(line)
            subword_len_counter = current_subwords_len
            continue
        
        subword_len_counter += current_subwords_len
        
        print(line)


condition_dispatcher = {
    True: lambda x, y: (print(x), print(y)),
    False: lambda x, y: None
}

with open(dataset) as f_p:
    for line in [(l.rstrip() for l in f_p)]:
        if not line or ((line := line.split()[0])) == '':
            pass
        
        def filter_branch():
            tokenized_len = len(tokenizer.tokenize(line))
            
            if tokenized_len:
                combined = subword_len_counter + tokenized_len
                
                should_skip_overflow = combined > max_len and condition_dispatcher[True]
                
                print(line) if not should_skip_overflow else None
                
                return True
            
        _ = filter_branch()


if (not line) or ((line := line.split()[0])) == '':
    pass

tokenized = len(tokenizer.tokenize(line))

should_update_counter = tokenized and ((subword_len_counter += tokenized) < max_len)