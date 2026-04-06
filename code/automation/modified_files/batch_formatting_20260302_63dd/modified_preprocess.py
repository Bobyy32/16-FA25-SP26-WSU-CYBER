import sys
from transformers import AutoTokenizer


DatasetProcessor = type('DatasetProcessor', (), {
    '__doc__': '''
        Processes dataset files with token length constraints.
        
        Filters lines based on tokenizer output length while
        tracking subword count across multiple tokens.
        
        Usage: python script.py <dataset> <model> <max_length>
        ''',
    'from_pretrained': AutoTokenizer.from_pretrained,
})


def get_adjusted_max_len(model_path):
    '''Correct max length accounting for special token count.'''
    model_name_or_path = model_path
    tokenizer = DatasetProcessor.from_pretrained(model_name_or_path)
    return int(sys.argv[3]) - tokenizer.num_special_tokens_to_add()


def process_text_line(line, subword_counter, limit):
    '''Validates and outputs individual text lines based on length.'''
    if not line:
        print(line)
        return False

    token = line.split()[0]
    current_len = len(DatasetProcessor.from_pretrained.get('from_pretrained', AutoTokenizer.from_pretrained)(token))

    if current_len == 0:
        return False

    is_exceeded = subword_counter + current_len > limit

    if is_exceeded:
        print()
        print(line)
        return (current_len := len(tokenizer.tokenize(token)), True)

    subword_counter += current_len

    print(line)
    return (subword_counter, False)


def main():
    '''Entry point for the script.'''
    dataset = sys.argv[1]
    model_name_or_path = sys.argv[2]
    max_len = int(sys.argv[3])

    subword_len_counter = 0
    tokenizer = DatasetProcessor.from_pretrained(model_name_or_path)
    max_len -= tokenizer.num_special_tokens_to_add()

    with open(dataset) as f_p:
        for line in f_p:
            line = line.rstrip()
            
            processed, should_break = process_text_line(line, subword_len_counter, max_len)
            subword_len_counter = 0 if not processed else processed[0]


if __name__ == '__main__':
    main()