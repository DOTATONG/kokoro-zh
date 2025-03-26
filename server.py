from flask import Flask, Response, request, jsonify
from typing import Optional
from kokoro import KModel, KPipeline
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import json


VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
CUDA_AVAILABLE = torch.cuda.is_available()
REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"
SAMPLE_RATE = 24000
VARS_VOICES = {
    "af_maple",
    "af_sol",
    "bf_vale",
    "zf_001",
    "zf_002",
    "zf_003",
    "zf_004",
    "zf_005",
    "zf_006",
    "zf_007",
    "zf_008",
    "zf_017",
    "zf_018",
    "zf_019",
    "zf_021",
    "zf_022",
    "zf_023",
    "zf_024",
    "zf_026",
    "zf_027",
    "zf_028",
    "zf_032",
    "zf_036",
    "zf_038",
    "zf_039",
    "zf_040",
    "zf_042",
    "zf_043",
    "zf_044",
    "zf_046",
    "zf_047",
    "zf_048",
    "zf_049",
    "zf_051",
    "zf_059",
    "zf_060",
    "zf_067",
    "zf_070",
    "zf_071",
    "zf_072",
    "zf_073",
    "zf_074",
    "zf_075",
    "zf_076",
    "zf_077",
    "zf_078",
    "zf_079",
    "zf_083",
    "zf_084",
    "zf_085",
    "zf_086",
    "zf_087",
    "zf_088",
    "zf_090",
    "zf_092",
    "zf_093",
    "zf_094",
    "zf_099",
    "zm_009",
    "zm_010",
    "zm_011",
    "zm_012",
    "zm_013",
    "zm_014",
    "zm_015",
    "zm_016",
    "zm_020",
    "zm_025",
    "zm_029",
    "zm_030",
    "zm_031",
    "zm_033",
    "zm_034",
    "zm_035",
    "zm_037",
    "zm_041",
    "zm_045",
    "zm_050",
    "zm_052",
    "zm_053",
    "zm_054",
    "zm_055",
    "zm_056",
    "zm_057",
    "zm_058",
    "zm_061",
    "zm_062",
    "zm_063",
    "zm_064",
    "zm_065",
    "zm_066",
    "zm_068",
    "zm_069",
    "zm_080",
    "zm_081",
    "zm_082",
    "zm_089",
    "zm_091",
    "zm_095",
    "zm_096",
    "zm_097",
    "zm_098",
    "zm_100",
}


def _is_true(value: Optional[str]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return value.upper() in VARS_TRUE_VALUES


def _as_float(value: Optional[str] | Optional[int]) -> float:
    if value is None:
        return 1
    return float(value)


def _is_voice(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.upper() in VARS_VOICES


out_path = Path(__file__).parent
cpu_model = KModel(repo_id=REPO_ID).to("cpu").eval()
gpu_model = KModel(repo_id=REPO_ID).to("cuda").eval() if CUDA_AVAILABLE else None

en_pipeline = KPipeline(lang_code="a", repo_id=REPO_ID, model=False)


def en_callable(text):
    if text == "Kokoro":
        return "kˈOkəɹO"
    elif text == "Sol":
        return "sˈOl"
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


zh_pipeline = KPipeline(
    lang_code="z", repo_id=REPO_ID, model=False, en_callable=en_callable
)


# 生成语音
def makeVoice(text, voice="zf_001", speed=1, use_gpu=CUDA_AVAILABLE):
    pack = zh_pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    audio_chunks = []
    for _, ps, _ in zh_pipeline(text, voice, speed):
        ref_s = pack[len(ps) - 1]
        if use_gpu:
            audio = gpu_model(ps, ref_s, speed)
        else:
            audio = cpu_model(ps, ref_s, speed)
        audio_chunks.append(audio.numpy())
    return np.concatenate(audio_chunks)


# 流式生成语音
def makeStream(text, voice="zf_001", speed=1, use_gpu=CUDA_AVAILABLE):
    pack = zh_pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    first = True
    for _, ps, _ in zh_pipeline(text, voice, speed):
        ref_s = pack[len(ps) - 1]
        if use_gpu:
            audio = gpu_model(ps, ref_s, speed)
        else:
            audio = cpu_model(ps, ref_s, speed)
        yield audio.numpy()
        if first:
            first = False
            yield torch.zeros(1).numpy()


app = Flask(__name__)


def output(data, status=200):
    return app.response_class(
        response=json.dumps(data, ensure_ascii=False),
        status=status,
        mimetype="application/json",
    )


@app.route("/api/voices", methods=["GET"])
def getVoices():
    return output({"success": True, "data": list(VARS_VOICES), "message": "获取成功"})


@app.route("/api/generate", methods=["POST"])
def generate():
    if not request.form:
        return output({"success": False, "message": "请求必须是表单形式"}, 400)

    # 获取表单数据
    text = request.form.get("text")
    voice = request.form.get("voice", "zf_001")
    speed = request.form.get("speed", 1)
    use_gpu = request.form.get("gpu", "true")
    stream = request.form.get("stream", "false")

    # 简单的数据验证
    if not text:
        return output({"success": False, "message": "缺少文本内容"}, 400)

    voice = voice if _is_voice(voice) else "zf_001"
    speed = _as_float(speed)
    use_gpu = _is_true(use_gpu)
    use_stream = _is_true(stream)

    response_data = {"success": True, "message": "生成成功", "data": {}}

    try:
        if use_stream:

            def generate_audio():
                for audio_chunk in makeStream(text, voice, speed, use_gpu):
                    yield audio_chunk.tobytes()

            return Response(generate_audio(), content_type="audio/wav")
        else:
            raw = makeVoice(text, voice, speed, use_gpu)
            f = out_path / f"voice.wav"
            sf.write(f, raw, SAMPLE_RATE)
            response_data["data"]["filepath"] = str(f)
    except Exception as e:
        response_data["success"] = False
        response_data["message"] = f"生成语音文件时出错: {str(e)}"

    return output(response_data)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
