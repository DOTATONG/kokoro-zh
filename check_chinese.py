# This file is hardcoded to transparently reproduce HEARME_zh.wav
# Therefore it may NOT generalize gracefully to other texts
# Refer to Usage in README.md for more general usage patterns

# pip install kokoro>=0.8.1 "misaki[zh]>=0.8.1"
from kokoro import KModel, KPipeline
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'
SAMPLE_RATE = 24000
VOICE = 'zf_001' if True else 'zm_010'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


en_pipeline = KPipeline(lang_code='a', repo_id=REPO_ID, model=False)
def en_callable(text):
    if text == 'Kokoro':
        return 'kˈOkəɹO'
    elif text == 'Sol':
        return 'sˈOl'
    return next(en_pipeline(text)).phonemes

# HACK: Mitigate rushing caused by lack of training data beyond ~100 tokens
# Simple piecewise linear fn that decreases speed as len_ps increases
def speed_callable(len_ps):
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1

zh_model = KModel(repo_id=REPO_ID).to(device).eval()
zh_pipeline = KPipeline(lang_code='z', repo_id=REPO_ID, model=zh_model, en_callable=en_callable)


text = "二愣子姓韩名立，这么像模像样的名字,他父母可起不出来，这是他父亲用两个粗粮制成的窝头，求村里老张叔给起的名字。"


generator = zh_pipeline(text, voice=VOICE, speed=speed_callable)

path = Path(__file__).parent
f = path / f'voice.wav'
result = next(generator)
wav = result.audio
sf.write(f, wav, SAMPLE_RATE)
