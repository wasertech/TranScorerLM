import torch, torchaudio
import contextlib
import wave
import numpy as np
import time

from transformers import (
        Wav2Vec2Processor,
        Wav2Vec2ForCTC
    )

class TranScorer:
    
    def __init__(self, accoustic_model_name, language_model_name, **kwargs):
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = time.perf_counter()
        print("Loading TranScorer...", end="")
        self.sample_rate = 16_000 # Hz
        self.accoustic_processor = Wav2Vec2Processor.from_pretrained(accoustic_model_name)
        self.language_model = Wav2Vec2ForCTC.from_pretrained(language_model_name)
        _t = time.perf_counter()
        print(f"...Took {_t - t} second(s).")

    def transcribe(self, audio_file_path):
        t = time.perf_counter()
        print(f"Loading {audio_file_path}...", end="")
        wav = self.read_wav(audio_file_path)
        _t = time.perf_counter()
        print(f"...Took {_t - t} second(s).")
        
        t = time.perf_counter()
        print(f"Tokenizing...", end="")
        vec = self.wav2vec(wav)
        _t = time.perf_counter()
        print(f"...Took {_t - t} second(s).")
        
        t = time.perf_counter()
        print(f"Decoding speech...", end="")
        txt = self.vec2txt(vec)
        _t = time.perf_counter()
        print(f"...Took {_t - t} second(s).")
        
        return txt[0]
    
    def read_wav(self, wav_path):
        # with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
        #     b = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.float32)
        #     return [b[~np.isnan(b)]]
        wav, sample_rate = torchaudio.load(wav_path)

        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)

        return wav[0]
    
    def wav2vec(self, wav):
        return self.tokenize_wav(wav)

    def vec2txt(self, vec):
        logits = self.get_logits(vec)
        predict = self.predict_argmax(logits)
        return self.decode_transcript_predict(predict)

    def tokenize_wav(self, wav):
        return self.accoustic_processor(
                    wav,
                    return_tensors="pt",
                    padding="longest",
                    sampling_rate=self.sample_rate,
                ).input_values
    
    def get_logits(self, vec):
        return self.language_model(vec).logits

    def predict_argmax(self, logits):
        return torch.argmax(logits, dim=-1)

    def decode_transcript_predict(self, predicted_argmax):
        return self.accoustic_processor.batch_decode(predicted_argmax)
