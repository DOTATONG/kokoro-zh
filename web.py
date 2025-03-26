from kokoro import KModel, KPipeline
import gradio as gr
import numpy as np
import torch

REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'
CUDA_AVAILABLE = torch.cuda.is_available()
print(torch.__version__)

cpu_model = KModel(repo_id=REPO_ID).to('cpu').eval()
gpu_model = KModel(repo_id=REPO_ID).to('cuda').eval()

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


zh_pipeline = KPipeline(lang_code='z', repo_id=REPO_ID, model=False, en_callable=en_callable)

CHOICES = {
    'af_maple': 'af_maple',
    'af_sol': 'af_sol',
    'bf_vale': 'bf_vale',
    'zf_001': 'zf_001',
    'zf_002': 'zf_002',
    'zf_003': 'zf_003',
    'zf_004': 'zf_004',
    'zf_005': 'zf_005',
    'zf_006': 'zf_006',
    'zf_007': 'zf_007',
    'zf_008': 'zf_008',
    'zf_017': 'zf_017',
    'zf_018': 'zf_018',
    'zf_019': 'zf_019',
    'zf_021': 'zf_021',
    'zf_022': 'zf_022',
    'zf_023': 'zf_023',
    'zf_024': 'zf_024',
    'zf_026': 'zf_026',
    'zf_027': 'zf_027',
    'zf_028': 'zf_028',
    'zf_032': 'zf_032',
    'zf_036': 'zf_036',
    'zf_038': 'zf_038',
    'zf_039': 'zf_039',
    'zf_040': 'zf_040',
    'zf_042': 'zf_042',
    'zf_043': 'zf_043',
    'zf_044': 'zf_044',
    'zf_046': 'zf_046',
    'zf_047': 'zf_047',
    'zf_048': 'zf_048',
    'zf_049': 'zf_049',
    'zf_051': 'zf_051',
    'zf_059': 'zf_059',
    'zf_060': 'zf_060',
    'zf_067': 'zf_067',
    'zf_070': 'zf_070',
    'zf_071': 'zf_071',
    'zf_072': 'zf_072',
    'zf_073': 'zf_073',
    'zf_074': 'zf_074',
    'zf_075': 'zf_075',
    'zf_076': 'zf_076',
    'zf_077': 'zf_077',
    'zf_078': 'zf_078',
    'zf_079': 'zf_079',
    'zf_083': 'zf_083',
    'zf_084': 'zf_084',
    'zf_085': 'zf_085',
    'zf_086': 'zf_086',
    'zf_087': 'zf_087',
    'zf_088': 'zf_088',
    'zf_090': 'zf_090',
    'zf_092': 'zf_092',
    'zf_093': 'zf_093',
    'zf_094': 'zf_094',
    'zf_099': 'zf_099',
    'zm_009': 'zm_009',
    'zm_010': 'zm_010',
    'zm_011': 'zm_011',
    'zm_012': 'zm_012',
    'zm_013': 'zm_013',
    'zm_014': 'zm_014',
    'zm_015': 'zm_015',
    'zm_016': 'zm_016',
    'zm_020': 'zm_020',
    'zm_025': 'zm_025',
    'zm_029': 'zm_029',
    'zm_030': 'zm_030',
    'zm_031': 'zm_031',
    'zm_033': 'zm_033',
    'zm_034': 'zm_034',
    'zm_035': 'zm_035',
    'zm_037': 'zm_037',
    'zm_041': 'zm_041',
    'zm_045': 'zm_045',
    'zm_050': 'zm_050',
    'zm_052': 'zm_052',
    'zm_053': 'zm_053',
    'zm_054': 'zm_054',
    'zm_055': 'zm_055',
    'zm_056': 'zm_056',
    'zm_057': 'zm_057',
    'zm_058': 'zm_058',
    'zm_061': 'zm_061',
    'zm_062': 'zm_062',
    'zm_063': 'zm_063',
    'zm_064': 'zm_064',
    'zm_065': 'zm_065',
    'zm_066': 'zm_066',
    'zm_068': 'zm_068',
    'zm_069': 'zm_069',
    'zm_080': 'zm_080',
    'zm_081': 'zm_081',
    'zm_082': 'zm_082',
    'zm_089': 'zm_089',
    'zm_091': 'zm_091',
    'zm_095': 'zm_095',
    'zm_096': 'zm_096',
    'zm_097': 'zm_097',
    'zm_098': 'zm_098',
    'zm_100': 'zm_100',
}
for v in CHOICES.values():
    zh_pipeline.load_voice(v)


def generate_first(text, voice='zf_001', speed=1, use_gpu=CUDA_AVAILABLE):
    pack = zh_pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    audio_chunks = []
    for _, ps, _ in zh_pipeline(text, voice, speed):
        ref_s = pack[len(ps) - 1]
        try:
            if use_gpu:
                audio = gpu_model(ps, ref_s, speed)
            else:
                audio = cpu_model(ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Retrying with CPU. To avoid this error, change Hardware to CPU.')
                audio = cpu_model(ps, ref_s, speed)
            else:
                raise gr.Error(e)
        audio_chunks.append(audio.numpy())
    return (24000, np.concatenate(audio_chunks)), ps


# Arena API
def predict(text, voice='zf_001', speed=1):
    return generate_first(text, voice, speed, use_gpu=False)[0]


def tokenize_first(text, voice='zf_001'):
    for _, ps, _ in zh_pipeline(text, voice):
        return ps
    return ''


def generate_all(text, voice='zf_001', speed=1, use_gpu=CUDA_AVAILABLE):
    pack = zh_pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    first = True
    for _, ps, _ in zh_pipeline(text, voice, speed):
        ref_s = pack[len(ps) - 1]
        try:
            if use_gpu:
                audio = gpu_model(ps, ref_s, speed)
            else:
                audio = cpu_model(ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Retrying with CPU. To avoid this error, change Hardware to CPU.')
                audio = cpu_model(ps, ref_s, speed)
            else:
                raise gr.Error(e)
        yield 24000, audio.numpy()
        if first:
            first = False
            yield 24000, torch.zeros(1).numpy()


with gr.Blocks() as generate_tab:
    out_audio = gr.Audio(label='输出音频', interactive=False, streaming=False, autoplay=True)
    generate_btn = gr.Button('Generate', variant='primary')
    with gr.Accordion('输出Tokens', open=True):
        out_ps = gr.Textbox(interactive=False, show_label=False,
                            info='用于生成音频的Tokens，最多510个上下文长度。')
        tokenize_btn = gr.Button('Tokenize', variant='secondary')
        predict_btn = gr.Button('Predict', variant='secondary')

with gr.Blocks() as stream_tab:
    out_stream = gr.Audio(label='输出音频流', interactive=False, streaming=True, autoplay=True)
    with gr.Row():
        stream_btn = gr.Button('Stream', variant='primary')
        stop_btn = gr.Button('Stop', variant='stop')
    with gr.Accordion('Note', open=True):
        gr.DuplicateButton()

API_OPEN = True
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label='输入文字', info=f"支持任意多个字符，长文本需要换行分割。单行最多500个字符。")
            with gr.Row():
                voice = gr.Dropdown(list(CHOICES.items()), value='zf_001', label='Voice', info='语音和质量各有差异')
                use_gpu = gr.Dropdown(
                    [('ZeroGPU 🚀', True), ('CPU 🐌', False)],
                    value=CUDA_AVAILABLE,
                    label='硬件',
                    info='通常GPU推理更快, 但有利用率限制',
                    interactive=CUDA_AVAILABLE
                )
            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='Speed')

        with gr.Column():
            gr.TabbedInterface([generate_tab, stream_tab], ['Generate', 'Stream'])

    generate_btn.click(fn=generate_first, inputs=[text, voice, speed, use_gpu], outputs=[out_audio, out_ps])
    tokenize_btn.click(fn=tokenize_first, inputs=[text, voice], outputs=[out_ps])
    stream_event = stream_btn.click(fn=generate_all, inputs=[text, voice, speed, use_gpu], outputs=[out_stream])
    stop_btn.click(fn=None, cancels=stream_event)
    predict_btn.click(fn=predict, inputs=[text, voice, speed], outputs=[out_audio])

if __name__ == '__main__':
    app.queue(api_open=API_OPEN).launch(server_name="0.0.0.0", server_port=40001, show_api=API_OPEN)
