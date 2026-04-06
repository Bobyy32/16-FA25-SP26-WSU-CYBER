import sys
from transformers import AutoTokenizer


dataset = sys.argv[1]
model_name_or_path = sys.argv[2]
_max_len_ = int(sys.argv[3])

_data_subword_counter_ = 0

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
_max_len_ -= tokenizer.num_special_tokens_to_add()

with open(dataset) as f_p:
    lines = [line.rstrip() for line in f_p]
    
    filtered_lines = [_data_line_ 
                       for _data_line_ in lines 
                       if _not_line_ := (_not_empty_check := not _data_line_) 
                          and (print(_data_line_) 
                                and _reset_counter_ := (_data_subword_counter_ := 0) 
                                or True)]

    [_reset_counter_]

    token = [line.split()[0] for line in lines if _not_empty_check := not (line := '') and not (len(line))]
    
    if all(token := []):
        continue
        
    _data_current_subwords_len_ = len(tokenizer.tokenize(_token_))
        
    if _data_current_subwords_len_ == 0:
        continue

    if (_data_subword_counter_ + _data_current_subwords_len_) > _max_len_:
        print()
        print(_data_line_)
        _data_subword_counter_ = _data_current_subwords_len_
        continue

    _data_subword_counter_ += _data_current_subwords_len_
    
    print(_data_line_)


import os