# TranScorerLM

Transformer as Scorer (Language Model) for STT accoustic models.

## Get started

```zsh
# Install with pip
❯ pip install git+https://github.com/wasertech/TranScorerLM.git

# Use TranScorer to convert an accoustic representation to text
❯ transcorer -f 'audio.wav'
Some weights of Wav2Vec2ForCTC were not initialized from the model checkpoint at facebook/wav2vec2-base-960h and are newly initialized: ['wav2vec2.masked_spec_embed']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Loading TranScorer......Took 2.2624789050023537 second(s).
Loading audio.wav......Took 0.00048493899521417916 second(s).
Tokenizing......Took 0.0008750690030865371 second(s).
Decoding speech......Took 0.21528533000673633 second(s).
CAN I TEST YOU
```

When `--file` is a valid audio file, it is used to get the accoustic representation.

You can also use the *`scorer`* `python` module.

```python
from pathlib import Path

from scorer.transformer import TranScorer

ts = TranScorer("path/to/scorer.transform")

to_transcribe = "path/to/audio.wav" 
transcript = ts.transcribe(to_transcribe)
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
    --do_train y \
    --do_eval y \
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
    --seed 0
```

You can also pass a `--push_to_hub` flag along with `--hub_token '${HUB_API_TOKEN}' --push_to_hub_model_id '${MODEL_ID}' --use_auth_token y`; where `$HUB_API_TOKEN` is a valid HuggingFace API Token (with write access) and `$MODEL_ID` is the name of the repo you want to create for your model, this will publish it on the hub under the above mentionned repository.

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
