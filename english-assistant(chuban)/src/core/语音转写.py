import whisper
import sounddevice as sd
from scipy.io.wavfile import write

def record_audio1(duration=5, fs=16000):
    """录音并返回音频数据"""
    print("\n▶ 开始录音... 请保持安静")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # 等待录音完成
    write("temp_recording.wav", fs, audio)  # 保存临时文件
    return audio, fs

def transcribe_audio(audio_path="temp_recording.wav"):
    """使用Whisper进行语音转写"""
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    return result["text"]