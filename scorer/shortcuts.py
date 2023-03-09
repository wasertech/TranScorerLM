import torch
import numpy as np

from scorer.transformer.tokenizer import load_tokenizer, tokenize

def is_cuda_available():
    return torch.cuda.is_available()

def transcode_stt(speech):
    '''
    Transcode from Speech-To-Text.
    '''
    text = "Text"
    return text

def transcode_tts(text):
    '''
    Transcode from Text-To-Speech.
    '''
    speech = None
    return speech

def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)

def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]