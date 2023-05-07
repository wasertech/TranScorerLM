# Dataset Importer for DeepSpeech/STT CSV dataset format

## wav_filename, wav_filesize, transcript

import os
from pathlib import Path
#from glob import glob
from datasets import concatenate_datasets, load_dataset, DatasetDict
from datasets.features import Audio


def load_16khz_audio(dataset):
    sr = 16000
    audio_column_name = "wav_filename"
    return dataset.cast_column(
            audio_column_name, Audio(sampling_rate=sr)
        )

def load_train_dataset_csv(data_path):
    pass

def load_dev_dataset_csv(data_path):
    pass

def load_test_dataset_csv(data_path, max_eval_samples=None, text_column_name="transcript"):
    #test_files = glob(f"{str(data_path)}/**/*_dev.csv")

    test_files = Path(data_path).rglob('*_dev.csv')
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

        # Filter raw_datasets['eval'] to only include row with transcript not None
        test_data["test"] = test_data["train"].filter(lambda row: row[text_column_name] not in [None, "", " ", "\n"])

        dataset_dict[f"{_dn.lower().replace(' ', '_')}"] = load_16khz_audio(test_data)
    
    return dataset_dict

def load_dataset_csv(data_path):
    ds = DatasetDict()
    ds["test"] = load_test_dataset_csv(data_path)