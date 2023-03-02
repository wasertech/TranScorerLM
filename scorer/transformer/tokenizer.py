# ! pip install tokenizers
import sys

from pathlib import Path
from argparse import ArgumentParser

from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

def tokenize(language, data_path):
    '''
    Produces a tokenizer for any given
    -   data_path: path to dir containing txt file to tokenize
    -   language: name of the language to save the tokenizer as

    Creates vocab.json and merges.txt
    # vocab.json
    {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "!": 5,
        "\"": 6,
        "#": 7,
        "$": 8,
        "%": 9,
        "&": 10,
        "'": 11,
        "(": 12,
        ")": 13,
        # ...
    }

    # merges.txt
    l a
    Ġ k
    o n
    Ġ la
    t a
    Ġ e
    Ġ d
    Ġ p
    # ...
    '''

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=[str(x) for x in Path(data_path).glob("*.txt")], vocab_size=500_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer.save_model(".", language)

def load_tokenizer(
        tokenizer_path,
        processing_step=[
            "</s>",
            "<s>",
        ],
        truncate_from=512,
    ):
    '''
    Land and return a tokenizer given its path
    '''
    tokenizer = ByteLevelBPETokenizer(
            f"{tokenizer_path}/vocab.json",
            f"{tokenizer_path}/merges.txt",
        )

    tokenizer._tokenizer.post_processor = BertProcessing(
            (t, tokenizer.token_to_id(t)) for t in processing_step
        )

    tokenizer.enable_truncation(max_length=truncate_from)
    
    return tokenizer


def get_tokens(sentence: str, tokenizer):
    '''
    Use a give tokenizer to tokenize a given sentence.
    '''
    return tokenizer.encode(sentence)

def parse_arguments():
    '''
    Parse arguments from the command line.
    '''

    arg_parser = ArgumentParser(description="Tokenize sentences.")
    arg_parser.add_argument(
            '-l',
            '--lang',
            type=str,
            default="generic_tokenizer",
            help="Language name to save the tokenizer as."
        )
    arg_parser.add_argument(
            '-d',
            '--data_path', 
            type=str,
            default=".",
            help="Language name to save the tokenizer as."
        )
    arg_parser.add_argument(
            '-t',
            '--test', 
            type=str,
            default="This is a simple test.",
            help="Text to tokenize as a test at the end."
        )

    return arg_parser.parse_args()

def main():
    '''
    Create a tokenizer given data and load it 
    '''
    args = parse_arguments()

    tokenize(args.lang, args.data_path)
    tokenizer = load_tokenizer(args.data_path)
    print(get_tokens(args.test, tokenizer))

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(1)
