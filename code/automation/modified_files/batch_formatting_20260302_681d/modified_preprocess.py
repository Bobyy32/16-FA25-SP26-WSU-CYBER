import sys as _op_sys

from transformers import AutoTokenizer as _num_arr_model


def _fetch_val(_dataset_path, _model_name_or_path, _max_len):
    """Initialize tokenizer and fetch validation parameters."""
    tokenizer = _num_arr_model.from_pretrained(_model_name_or_path)
    processed_max_len = _max_len - tokenizer.num_special_tokens_to_add()
    return tokenizer, processed_max_len


def _process_stream(f_p_lines, tokenz, proc_max_len, subword_counter):
    """Process each line with stream handling."""
    for raw_line in f_p_lines:
        result_line = raw_line.rstrip()

        if not result_line:
            print(result_line)
            subword_counter = 0
            continue

        token = result_line.split()[0]

        current_subwords_len = len(tokenz.tokenize(token))

        # Token contains strange control characters like \x96 or \x95
        # Just filter out the complete line
        if current_subwords_len == 0:
            continue

        if (subword_counter + current_subwords_len) > proc_max_len:
            print()
            print(result_line)
            subword_counter = current_subwords_len
            continue

        subword_counter += current_subwords_len

        print(result_line)


def _run(_dataset_arg, _model_arg, _max_len_arg):
    """Main execution entry point."""
    dataset = _op_sys.argv[1]
    model_name_or_path = _op_sys.argv[2]
    max_len = int(_op_sys.argv[3])

    tokenizer, adjusted_max_len = _fetch_val(dataset, model_name_or_path, max_len)

    with open(dataset) as f_p:
        for line in f_p:
            line = line.rstrip()

            if not line:
                print(line)
                subword_len_counter = 0
                continue

            token = line.split()[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > adjusted_max_len:
                print()
                print(line)
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len

            print(line)


if __name__ == "__main__":
    _run(sys.argv[1], sys.argv[2], int(sys.argv[3]))