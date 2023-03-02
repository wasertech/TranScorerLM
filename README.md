# TranScorerLM

Transformer as Scorer (Language Model) for STT accoustic models.

## Get started

```zsh
# Install with pip
pip install git+https://github.com/wasertech/TranScorerLM.git

# Use TranScorer to convert an accoustic representation to text
transcorer \
    --accoustic path/to/stt/accoustic/model/output_graph.tflite \
    --in_put (path/to/audio.wav|"Input text.") \
    --scorer path/to/stt/language/model/scorer.transform \
    --out_put path/to/output.txt
```

When `--in_put` is a valid audio file, it is used to get the accoustic representation.

Otherwise, accoustic representation is generated from text using TTS (VTCK/VITS). You can give a list of speakers to use (all by default).

You can also use the *`scorer`* `python` module.

```python
from pathlib import Path

from scorer import Scorer

# Import STT Transcoder from shortcuts
# You could transcode from Speech-To-Text (transcode_stt)
# or from Text-To-Speech (transcode_tts)
from scorer.shortcuts import transcode_stt

lm = Scorer("path/to/stt/language/model/scorer.transform")

to_transcribe = "path/to/audio.wav" 
transcript = transcode_stt("path/to/stt/accoustic/model/output_graph.tflite", to_transcribe if Path(to_transcribe).exists() else "Input text.", lm)
print(f"{to_transcribe=} -> {transcript=}")
```

You could also ask the scorer to tokenize something for you.

```
python -m scorer.transformer.tokenizer \
    --lang "english" \
    --data_path "parent/path/to/tokenizer/data/*.txt" \
    --test "This test sentence will be tokenized."
```

## Training a new scorer from scratch

Distribute `trainscorer`.

```zsh
# trainscorer -> python -m scorer.train
python -m trainer.distribute \
    --script trainscorer \
    --gpus "0,1" \
    --output_dir ./models/TranScorer-en \
    --model_type roberta \
    --mlm \
    --config_name ./models/TranScorer-en \
    --tokenizer_name ./models/TranScorer-en \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_gpu_train_batch_size 64 \
    --evaluate_during_training \
    --seed 42
```

You can also train directly from a `python` script.

```python
from scorer import TranScorerModelConfig, TranScorerModel, is_cuda_available

from trainer import Trainer
from trainer.trainer import TrainerArgs

def train(trainer):
    return trainer.fit()

def main():
    args = TrainerArgs()

    config = TranScorerModelConfig()
    config.batch_size = 64
    config.grad_clip = None

    model = TranScorerModel()

    is_cuda = is_cuda_available()

    trainer = Trainer(
            args,
            config,
            model=model,
            output_path=os.getcwd(),
            gpu=0 if is_cuda else None
        )

    trainer.config.epochs = 10
    
    return train(trainer)

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(1)
```

Training will create a custom tokenizer using availible sentences.

## License

This project is distributed under [Mozilla Public License 2.0](LICENSE).

## Contribute

Please read [CONTRIBUTE](CONTRIBUTE.md) before anthing.