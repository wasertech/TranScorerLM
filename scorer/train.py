#!/usr/bin/env python

""" Train a Transformers CTC model for automatic speech recognition """

import functools
import json
import logging
import os
import re
import sys
import warnings
from glob import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import datasets
import numpy as np
import torch
from datasets import DatasetDict, load_dataset, concatenate_datasets

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from scorer import evaluate
from scorer.transformer.dataset import TranScorerDataTrainingArguments, TrainingArguments, DataCollatorCTCWithPadding
from scorer.transformer.transcorer import TranScorerModelArguments

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.1")

require_version("datasets>=1.18.0", "To fix: pip install git+https://github.com/wasertech/TranScorerLM.git")


logger = logging.getLogger(__name__)

def create_vocabulary_from_data(
    datasets: DatasetDict,
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(batch):
        #print([key for key in batch.keys()])
        all_text = " ".join([text for text in batch["transcript"] if text is not None])
        vocab = list(set(all_text))
        #print(vocab[0], all_text[0])
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets["train"].column_names,
    )

    # take union of all unique characters in each dataset
    vocab_set = functools.reduce(
        lambda vocab_1, vocab_2: set(vocab_1["vocab"][0]) | set(vocab_2["vocab"][0]), vocabs.values()
    )

    vocab_set = [v for v in vocab_set if isinstance(v, str)]
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_set))}

    # replace white space with delimiter token
    if word_delimiter_token is not None and " " in vocab_dict:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict

def train():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((TranScorerModelArguments, TranScorerDataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 1. First, let's load the dataset
    wav2txt = DatasetDict()

    if training_args.do_train:
        train_files = glob(f"{str(data_args.data_path)}/**/*_train.csv")
        if not train_files:
            raise ValueError(f"No training files found under {data_args.data_path}")
        
        dataset_list = []
        for tf in train_files:
            train_data = load_dataset('csv', data_files=[tf])
            base_abs_path = os.path.abspath(os.path.dirname(tf))
            
            def get_absolute_wavpath(wavpath):
                wfn = wavpath['wav_filename']
                wfp = os.path.join(base_abs_path, wfn)
                wavpath['wav_filename'] = wfp
                return wavpath
            
            train_data = train_data.map(get_absolute_wavpath, desc="Mapping relative wav files to absolute path")

            wav2txt['train'] = train_data
        
        if data_args.audio_column_name not in train_data.column_names['train']:
            raise ValueError(
                f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'."
                " Make sure to set `--audio_column_name` to the correct audio column - one of"
                f" {', '.join(train_data.column_names['train'])}."
            )

        if data_args.text_column_name not in train_data.column_names['train']:
            raise ValueError(
                f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--text_column_name` to the correct text column - one of "
                f"{', '.join(train_data.column_names['train'])}."
            )

        if data_args.max_train_samples is not None:
            wav2txt["train"] = train_data.select(range(data_args.max_train_samples))
        


    if training_args.do_eval:
        dev_files = glob(f"{str(data_args.data_path)}/**/*_dev.csv")
        if not dev_files:
            raise ValueError(f"No dev files found under {data_args.data_path}")
        wav2txt['eval'] = load_dataset('csv', data_files=dev_files)
        eval_data = wav2txt['eval']

        if data_args.max_eval_samples is not None:
            wav2txt["eval"] = eval_data.select(range(data_args.max_eval_samples))

    # 2. We remove some special characters from the datasets
    # that make training complicated and do not help in transcribing the speech
    # E.g. characters, such as `,` and `.` do not really have an acoustic characteristic
    # that could be easily picked up by the model
    chars_to_ignore_regex = (
        f'[{"".join(data_args.chars_to_ignore)}]' if data_args.chars_to_ignore is not None else None
    )
    text_column_name = data_args.text_column_name

    def remove_special_characters(batch):
        if batch[text_column_name] is None:
            # Return the batch unchanged if text_column_name is None
            return batch

        # Process each sentence in the list separately
        if batch[text_column_name] is not None:
            # Apply the regular expression substitution to each sentence
            processed_sentence = re.sub(chars_to_ignore_regex, "", batch[text_column_name]).lower()
            batch[text_column_name] = processed_sentence

        return batch


    with training_args.main_process_first(desc="dataset map special characters removal"):
        wav2txt = train_data.map(
            remove_special_characters,
            # remove_columns=[text_column_name],
            desc="remove special characters from datasets",
        )

    # save special tokens for tokenizer
    word_delimiter_token = data_args.word_delimiter_token
    unk_token = data_args.unk_token
    pad_token = data_args.pad_token

    # 3. Next, let's load the config as we might need it to create
    # the tokenizer
    # load config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_auth_token=data_args.use_auth_token
    )

    # 4. Next, if no tokenizer file is defined,
    # we create the vocabulary of the model by extracting all unique characters from
    # the training and evaluation datasets
    # We need to make sure that only first rank saves vocabulary
    # make sure all processes wait until vocab is created
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    tokenizer_kwargs = {}
    if tokenizer_name_or_path is None:
        # save vocab in training output dir
        tokenizer_name_or_path = training_args.output_dir

        vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")

        with training_args.main_process_first():
            if training_args.overwrite_output_dir and os.path.isfile(vocab_file):
                try:
                    os.remove(vocab_file)
                except OSError:
                    # in shared file-systems it might be the case that
                    # two processes try to delete the vocab file at the some time
                    pass

        with training_args.main_process_first(desc="dataset map vocabulary creation"):
            if not os.path.isfile(vocab_file):
                os.makedirs(tokenizer_name_or_path, exist_ok=True)
                vocab_dict = create_vocabulary_from_data(
                    wav2txt,
                    word_delimiter_token=word_delimiter_token,
                    unk_token=unk_token,
                    pad_token=pad_token,
                )

                # save vocab dict to be loaded into tokenizer
                with open(vocab_file, "w") as file:
                    json.dump(vocab_dict, file)

        # if tokenizer has just been created
        # it is defined by `tokenizer_class` if present in config else by `model_type`
        tokenizer_kwargs = {
            "config": config if config.tokenizer_class is not None else None,
            "tokenizer_type": config.model_type if config.tokenizer_class is None else None,
            "unk_token": unk_token,
            "pad_token": pad_token,
            "word_delimiter_token": word_delimiter_token,
        }

    # 5. Now we can instantiate the feature extractor, tokenizer and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # load feature_extractor and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_auth_token=data_args.use_auth_token,
        **tokenizer_kwargs,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_auth_token=data_args.use_auth_token
    )

    # adapt config
    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": model_args.activation_dropout,
        }
    )

    # create model
    model = AutoModelForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        use_auth_token=data_args.use_auth_token,
    )

    # freeze encoder
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # 6. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`

    # make sure that dataset decodes audio with correct sampling rate
    dataset_sampling_rate = 16_000 #next(iter(wav2txt.values())).features[data_args.audio_column_name].sampling_rate
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        wav2txt = wav2txt.cast_column(
            data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )
    else:
        wav2txt = wav2txt.cast_column(
            data_args.audio_column_name, datasets.features.Audio(sampling_rate=dataset_sampling_rate)
        )
    # derive max & min input length for sample rate & max duration
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers

    # `phoneme_language` is only relevant if the model is fine-tuned on phoneme classification
    phoneme_language = data_args.phoneme_language

    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio
        sample = batch[audio_column_name]
        #print(sample)

        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])

        # encode targets
        additional_kwargs = {}
        if phoneme_language is not None:
            additional_kwargs["phonemizer_lang"] = phoneme_language

        batch["labels"] = tokenizer(batch["transcript"], **additional_kwargs).input_ids
        return batch

    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = wav2txt.map(
            prepare_dataset,
            remove_columns=next(iter(wav2txt.values())).column_names,
            num_proc=num_workers,
            desc="preprocess datasets",
        )

        def is_audio_in_length_range(length):
            return length > min_input_length and length < max_input_length

        # filter data that is shorter than min_input_length
        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_length"],
        )

    # 7. Next, we can prepare the training.
    # Let's use word error rate (WER) as our evaluation metric,
    # instantiate a data collator and the trainer

    # Define evaluation metrics during training, *i.e.* word error rate, character error rate
    eval_metrics = {metric: evaluate.load(metric) for metric in data_args.eval_metrics}

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}

        return metrics

    # Now save everything to be able to create a single processor later
    if is_main_process(training_args.local_rank):
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)

    try:
        processor = AutoProcessor.from_pretrained(training_args.output_dir)
    except (OSError, KeyError):
        warnings.warn(
            "Loading a processor from a feature extractor config that does not"
            " include a `processor_class` attribute is deprecated and will be removed in v5. Please add the following "
            " attribute to your `preprocessor_config.json` file to suppress this warning: "
            " `'processor_class': 'Wav2Vec2Processor'`",
            FutureWarning,
        )
        processor = Wav2Vec2Processor.from_pretrained(training_args.output_dir)

    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        tokenizer=feature_extractor,
    )

    # 8. Finally, we can start training

    # Training
    if training_args.do_train:
        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    config_name = data_args.dataset_config_name if data_args.dataset_config_name is not None else "na"
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
        "tags": ["automatic-speech-recognition", data_args.dataset_name],
        "dataset_args": (
            f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split:"
            f" {data_args.eval_split_name}"
        ),
        "dataset": f"{data_args.dataset_name.upper()} - {config_name.upper()}",
    }
    if "common_voice" in data_args.dataset_name:
        kwargs["language"] = config_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results

if __name__ == "__main__":
    try:
        train()
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(1)
