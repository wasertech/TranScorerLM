import torchaudio
from datasets import load_dataset, load_metric
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
import evaluate
import torch
import re


from scorer.dataset.csv import load_test_dataset_csv

model_name = "wasertech/wav2vec2-cv-fr-9"
device = "cuda"

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'  # noqa: W605

model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_name)

datasets = load_test_dataset_csv("/mnt/extracted/data")

def map_to_pred(batch):
    features = processor(batch["wav_filename"]['array'], sampling_rate=16000, padding=True, return_tensors="pt")
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch["predicted"] = processor.batch_decode(pred_ids)
    batch["target"] = batch["transcript"].apply(lambda x: re.sub(chars_to_ignore_regex, "", x))
    return batch


def main():
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")

    for split, data in datasets.items():

        print(f"Evalutating {split}...")

        result = data.map(map_to_pred, batched=True, batch_size=16, remove_columns=list(data['test'].features.keys()))

        _w = wer.compute(predictions=result["predicted"], references=result["target"])
        _c = cer.compute(predictions=result["predicted"], references=result["target"])
        
        print("-"*13)
        print(f"|\t{split}\t|")
        print("-"*13)
        print("|\tWER\t|\tCER\t|")
        print(f"|\t{_w['wer']:.2%}\t|\t{_c['cer']:.2%}\t|")
        print("-"*13)

    
    result = datasets['test'].map(map_to_pred, batched=True, batch_size=16, remove_columns=list(datasets['test'].features.keys()))

    _w = wer.compute(predictions=result["predicted"], references=result["target"])
    _c = cer.compute(predictions=result["predicted"], references=result["target"])
    
    print("-"*13)
    print(f"|\tAverage Metrics\t|")
    print("-"*13)
    print("|\tWER\t|\tCER\t|")
    print(f"|\t{_w['wer']:.2%}\t|\t{_c['cer']:.2%}\t|")
    print("-"*13)

if __name__ == "__main__":
    main()