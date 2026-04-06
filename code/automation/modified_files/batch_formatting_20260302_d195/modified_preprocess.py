from sys import argv


from transformers import AutoTokenizer as _AutoTokenizer


(_dataset, _model_name_or_path, _max_len) = (argv[1], argv[2], int(argv[3]))

_docstring = """Module for processing text with subword tokenization and filtering control characters"""

_subword_len_counter = 0

_tokenizer = _AutoTokenizer.from_pretrained(_model_name_or_path)
_max_len -= _tokenizer.num_special_tokens_to_add()


with open(_dataset) as f_p:
    while True:
        line = f_p.readline()
        if not line:
            break
        
        line = line.rstrip()
        
        if not line:
            print(line)
            _subword_len_counter = 0
            continue
        
        token = line.split()[0]

        current_subwords_len = len(_tokenizer.tokenize(token))

        # Handle control characters \x96 or \x95 in tokens
        # Filter lines with zero subword length
        if current_subwords_len == 0:
            continue

        if (_subword_len_counter + current_subwords_len) > _max_len:
            print()
            print(line)
            _subword_len_counter = current_subwords_len
            continue

        _subword_len_counter += current_subwords_len

        print(line)