# Dataset Importer for DeepSpeech/STT CSV dataset format

## wav_filename, wav_filesize, transcript

import os
from pathlib import Path
#from glob import glob
from datasets import concatenate_datasets, load_dataset, DatasetDict
from datasets.features import Audio


def load_16khz_audio(dataset, sr = 16000, audio_column_name = "wav_filename"):
    return dataset.cast_column(
            audio_column_name, Audio(sampling_rate=sr)
        )

def load_train_dataset_csv(data_path, max_eval_samples=None, text_column_name="transcript"):
    train_files = [ p.as_posix() for p in Path(data_path).rglob('*train.csv') ]
    if not train_files:
        raise ValueError(f"No test files found under {data_path}")

    dataset_dict = DatasetDict()
    for tf in train_files:
        train_data = load_dataset('csv', data_files=[tf])
        _dn = os.path.dirname(tf)
        base_abs_path = os.path.abspath(_dn)

        def get_absolute_wavpath(wavpath):
            wfn = wavpath['wav_filename']
            wfp = os.path.join(base_abs_path, wfn)
            wavpath['wav_filename'] = wfp
            return wavpath
        
        train_data = train_data.map(
            get_absolute_wavpath, desc="Mapping relative wav files to absolute path"
        )

        if max_eval_samples is not None:
            train_data["train"] = train_data["train"].select(range(max_eval_samples))

        # Filter raw_datasets['eval'] to only include row with transcript
        train_data["train"] = train_data["train"].filter(lambda row: row[text_column_name] not in [None, "", " ", "\n"])

        train_data["train"] = train_data["train"].rename_column("wav_filesize", "input_length")

        dataset_dict[f"{_dn.lower().replace(' ', '_')}"] = load_16khz_audio(train_data)
    
    return dataset_dict

def load_dev_dataset_csv(data_path, max_eval_samples=None, text_column_name="transcript"):
    eval_files = [ p.as_posix() for p in Path(data_path).rglob('*dev.csv') ]
    if not eval_files:
        raise ValueError(f"No test files found under {data_path}")

    dataset_dict = DatasetDict()
    for tf in eval_files:
        eval_data = load_dataset('csv', data_files=[tf])
        _dn = os.path.dirname(tf)
        base_abs_path = os.path.abspath(_dn)

        def get_absolute_wavpath(wavpath):
            wfn = wavpath['wav_filename']
            wfp = os.path.join(base_abs_path, wfn)
            wavpath['wav_filename'] = wfp
            return wavpath
        
        eval_data = eval_data.map(
            get_absolute_wavpath, desc="Mapping relative wav files to absolute path"
        )

        if max_eval_samples is not None:
            eval_data["train"] = eval_data["train"].select(range(max_eval_samples))

        # Filter raw_datasets['eval'] to only include row with transcript
        eval_data["eval"] = eval_data["train"].filter(lambda row: row[text_column_name] not in [None, "", " ", "\n"])

        eval_data["eval"] = eval_data["eval"].rename_column("wav_filesize", "input_length")

        dataset_dict[f"{_dn.lower().replace(' ', '_')}"] = load_16khz_audio(eval_data)
    
    return dataset_dict

def load_test_dataset_csv(data_path, max_eval_samples=None, text_column_name="transcript"):
    test_files = [ p.as_posix() for p in Path(data_path).rglob('*test.csv') ]
    if not test_files:
        raise ValueError(f"No test files found under {data_path}")

    dataset_dict = DatasetDict()
    for tf in test_files:
        test_data = load_dataset('csv', data_files=[tf])
        _dn = os.path.dirname(tf)
        base_abs_path = os.path.abspath(_dn)

        def get_absolute_wavpath(wavpath):
            wfn = wavpath['wav_filename']
            wfp = os.path.join(base_abs_path, wfn)
            wavpath['wav_filename'] = wfp
            return wavpath
        
        test_data = test_data.map(
            get_absolute_wavpath, desc="Mapping relative wav files to absolute path"
        )

        if max_eval_samples is not None:
            test_data["train"] = test_data["train"].select(range(max_eval_samples))

        # Filter raw_datasets['eval'] to only include row with transcript
        test_data["test"] = test_data["train"].filter(lambda row: row[text_column_name] not in [None, "", " ", "\n"])

        test_data["test"] = test_data["test"].rename_column("wav_filesize", "input_length")

        dataset_dict[f"{_dn.lower().replace(' ', '_')}"] = load_16khz_audio(test_data)
    
    return dataset_dict

def load_dataset_csv(data_path):
    ds = DatasetDict()
    ds["test"] = load_test_dataset_csv(data_path)