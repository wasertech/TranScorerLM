import torch

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

