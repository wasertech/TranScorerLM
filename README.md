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

Start training a scorer using  `trainscorer`.

```zsh
# trainscorer -> python -m scorer.train
trainscorer \
    --model_name_or_path facebook/wav2vec2-base \
    --dataset_name wav2txt \
    --dataset_config_name wav2txt \
    --output_dir wav2txt/models \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --fp16 \
    --learning_rate 3e-5 \
    --max_length_seconds 1 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 4 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --save_total_limit 3 \
    --seed 0 \
    --push_to_hub
```

You can also train directly from a `python` script.

```python
from scorer.train import train

if __name__ == "__main__":
    try:
        train()
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(1)
```

Training will create a custom tokenizer using availible sentences.

## License

This project is distributed under [Mozilla Public License 2.0](LICENSE).

## Contribute

Please read [CONTRIBUTE](CONTRIBUTE.md) before anthing.
