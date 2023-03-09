import sys
import argparse
from pathlib import Path

from scorer.transformer import TranScorer

def parse_arguments():
    parse = argparse.ArgumentParser(description="Transformer as CTC Scorer for STT.")
    parse.add_argument('-v', '--version', action='store_true', help="shows the current version of translator")
    parse.add_argument('-f', '--file', required=False, help='Path to the audio file to run (WAV format)')
    parse.add_argument('-am', '--accoustic_model', default="facebook/wav2vec2-base-960h", help="Model path or HuggingFace ID to use as the accoustic processor.")
    parse.add_argument('-lm', '--language_model', default="facebook/wav2vec2-base-960h", help="Model path or HuggingFace ID to use as the language model.")

    return parse.parse_args()

def main():
    args = parse_arguments()

    accoustic_model_name, language_model_name = args.accoustic_model, args.language_model

    if args.version:
        from scorer import __version__
        from transformers import __version__ as __trans_version__
        print(f"Scorer: {__version__}")
        print(f"Transformer: {__trans_version__}")
        sys.exit(0)

    wav_file = Path(args.file)
    if not wav_file.exists() and not wav_file.is_file():
        raise FileNotFoundError(wav_file)
    
    transcorer = TranScorer(accoustic_model_name, language_model_name)

    transcription = transcorer.transcribe(str(wav_file))

    print(transcription)

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(1)
