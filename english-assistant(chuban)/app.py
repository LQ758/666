import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
# 创建Flask应用实例
app = Flask(__name__)



# 提供静态文件访问
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# 导入核心功能模块
import sys
import os
import json

# 添加src目录到Python路径
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_dir)

# 直接导入核心模块，与main.py保持完全一致的导入方式
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./src'))

# 导入功能模块
from src.core.data_processing import load_sentences_and_paths, get_random_sentence
from src.core.发音评分模块 import record_audio, score_pronunciation, score_pronunciation_detailed
from src.core.语法检查 import analyze_grammar
from src.core.自定义练习模块 import load_custom_data, get_random_custom_sentence
from src.core.处理txt文档 import shuijizhongwen
from src.core.语音转写 import record_audio1, transcribe_audio
print('✅ 成功导入所有核心模块')

# 全局录音状态
is_recording = False

# 录音保存目录（集中保存到 data/audio/uploads 下）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_UPLOAD_DIR = os.path.join(BASE_DIR, 'data', 'audio', 'uploads')
os.makedirs(AUDIO_UPLOAD_DIR, exist_ok=True)
KEEP_UPLOADS = True  # 如需保留上传文件以便排查或回放，将其改为 True
print(f"🎯 音频上传目录: {AUDIO_UPLOAD_DIR}")

# 录音线程函数
def record_audio_thread():
    global is_recording
    try:
        record_audio1()
    finally:
        is_recording = False

@app.route('/')
def index():
    return render_template('index.html')  # 需要创建 HTML 文件

# 将多种音频格式转为标准 WAV 16k 单声道
def convert_to_wav_16k(input_path):
    try:
        import subprocess
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_16k.wav"
        
        # 检查ffmpeg是否可用
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("警告: ffmpeg未安装，无法转换音频格式")
            return None
        
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ac', '1', '-ar', '16000',
            '-f', 'wav',
            output_path
        ]
        
        print(f"执行音频转换命令: {' '.join(cmd)}")
        
        # 使用管道抑制输出
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"音频转换成功: {output_path}")
            return output_path
        else:
            print("音频转换失败: 输出文件不存在或为空")
            return None
    except Exception as e:
        print(f"音频转换过程中出错: {e}")
        return None

#随机英文句子接口
@app.route('/api/random-english-sentence', methods=['GET'])
def get_random_english_sentence():
    tsv_file = os.path.join("data", "common_voice", "validated.tsv")
    data_records = load_sentences_and_paths(tsv_file)
    random_record = get_random_sentence(data_records)
    reference_text = random_record["sentence"]
    return jsonify({"sentence": reference_text})
# 发音评分接口
@app.route('/api/score-pronunciation', methods=['POST'])
def score_pronunciation_api():
    try:
        print("=== 开始处理发音评分请求 ===")
        
        # 获取请求参数
        reference_text = request.form.get('reference_text')
        audio_file = request.files.get('audio_file')

        print(f"参考文本: {reference_text}")
        print(f"音频文件: {audio_file.filename if audio_file else 'None'}")

        # 参数验证
        if not reference_text:
            print("错误: 缺少参考文本")
            return jsonify({'error': '缺少参考文本'}), 400
        if not audio_file:
            print("错误: 缺少音频文件")
            return jsonify({'error': '缺少音频文件'}), 400

        # 生成唯一的临时文件路径
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        import os
        _, ext = os.path.splitext(audio_file.filename)
        ext = (ext or '.webm').lower()  # 默认使用webm格式
        audio_path = os.path.join(AUDIO_UPLOAD_DIR, f"temp_audio_{unique_id}{ext}")
        
        try:
            # 保存音频文件
            audio_file.save(audio_path)
            file_size = os.path.getsize(audio_path)
            print(f"音频文件已保存: {audio_path}, 大小: {file_size} 字节")

            # 检查文件是否成功保存
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print("错误: 音频文件保存失败或为空文件")
                return jsonify({"error": "音频文件保存失败或为空文件"}), 500

            # 如非 WAV，尝试转码为 16k WAV
            _, ext = os.path.splitext(audio_path)
            ext = (ext or '').lower()
            decoded_path = audio_path
            if ext != '.wav':
                print(f"音频格式为 {ext}，尝试转换为WAV格式...")
                converted = convert_to_wav_16k(audio_path)
                if converted:
                    decoded_path = converted
                    print(f"转换成功: {converted}")
                else:
                    print("转换失败，使用原始文件")
                    decoded_path = audio_path

            # 使用librosa读取音频文件转换为numpy数组
            try:
                import librosa
                print("正在加载音频文件...")
                
                # 尝试直接加载音频
                try:
                    audio_data, sr = librosa.load(decoded_path, sr=16000)
                    print(f"音频加载成功: 采样率={sr}, 长度={len(audio_data)}")
                except Exception as load_error:
                    print(f"直接加载失败: {load_error}")
                    
                    # 如果直接加载失败，尝试使用soundfile
                    try:
                        import soundfile as sf
                        print("尝试使用soundfile加载音频...")
                        audio_data, sr = sf.read(decoded_path)
                        if len(audio_data.shape) > 1:
                            audio_data = audio_data[:, 0]  # 取第一个声道
                        # 重采样到16kHz
                        if sr != 16000:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                        print(f"soundfile加载成功: 采样率={sr}, 长度={len(audio_data)}")
                    except Exception as sf_error:
                        print(f"soundfile加载也失败: {sf_error}")
                        raise RuntimeError(f"无法加载音频文件: {load_error}")
                
                # 检查音频数据
                if len(audio_data) == 0:
                    raise ValueError("音频数据为空")
                
                # 音频数据预处理
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # 音频归一化
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                print(f"音频预处理完成: 数据类型={audio_data.dtype}, 范围=[{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
                
            except Exception as e:
                print(f"音频加载失败: {e}")
                return jsonify({"error": f"音频加载失败: {str(e)}"}), 500

            # 调用核心评分函数
            try:
                print("开始调用发音评分函数...")
                print(f"音频数据: 长度={len(audio_data)}, 类型={audio_data.dtype}")
                print(f"参考文本: '{reference_text}'")
                
                score = score_pronunciation(audio_data, reference_text)
                print(f"评分完成: {score}")
                
                # 构建响应结果
                return jsonify({"score": f"{score:.1f}"})
            except Exception as e:
                print(f"发音评分函数调用失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 提供具体的错误信息
                error_msg = str(e)
                if "发音评分失败" in error_msg:
                    return jsonify({"error": error_msg}), 500
                else:
                    return jsonify({"error": f"评分计算失败: {error_msg}"}), 500
                
        finally:
            # 清理临时文件
            try:
                if os.path.exists(audio_path) and not KEEP_UPLOADS:
                    os.remove(audio_path)
                    print(f"临时文件已清理: {audio_path}")
                # 同时清理可能的转码产物
                base, _ = os.path.splitext(audio_path)
                converted = f"{base}_16k.wav"
                if os.path.exists(converted) and not KEEP_UPLOADS:
                    os.remove(converted)
                    print(f"转换文件已清理: {converted}")
            except Exception as cleanup_error:
                print(f"清理临时文件时出错: {str(cleanup_error)}")
    except Exception as e:
        print(f"发音评分接口错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"评分过程中出错: {str(e)}"}), 500

# 音素级发音评分接口
@app.route('/api/score-pronunciation-detailed', methods=['POST'])
def score_pronunciation_detailed_api():
    """音素级发音评分接口，返回详细的分析结果"""
    try:
        print("=== 开始处理音素级发音评分请求 ===")
        
        # 获取请求参数
        reference_text = request.form.get('reference_text')
        audio_file = request.files.get('audio_file')

        print(f"参考文本: {reference_text}")
        print(f"音频文件: {audio_file.filename if audio_file else 'None'}")

        # 参数验证
        if not reference_text:
            print("错误: 缺少参考文本")
            return jsonify({'error': '缺少参考文本'}), 400
        if not audio_file:
            print("错误: 缺少音频文件")
            return jsonify({'error': '缺少音频文件'}), 400

        # 生成唯一的临时文件路径
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        import os
        _, ext = os.path.splitext(audio_file.filename)
        ext = (ext or '.webm').lower()  # 默认使用webm格式
        audio_path = os.path.join(AUDIO_UPLOAD_DIR, f"temp_audio_detailed_{unique_id}{ext}")
        
        try:
            # 保存音频文件
            audio_file.save(audio_path)
            file_size = os.path.getsize(audio_path)
            print(f"音频文件已保存: {audio_path}, 大小: {file_size} 字节")

            # 检查文件是否成功保存
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print("错误: 音频文件保存失败或为空文件")
                return jsonify({"error": "音频文件保存失败或为空文件"}), 500

            # 如非 WAV，尝试转码为 16k WAV
            _, ext = os.path.splitext(audio_path)
            ext = (ext or '').lower()
            decoded_path = audio_path
            if ext != '.wav':
                print(f"音频格式为 {ext}，尝试转换为WAV格式...")
                converted = convert_to_wav_16k(audio_path)
                if converted:
                    decoded_path = converted
                    print(f"转换成功: {converted}")
                else:
                    print("转换失败，使用原始文件")
                    decoded_path = audio_path

            # 使用librosa读取音频文件转换为numpy数组
            try:
                import librosa
                print("正在加载音频文件...")
                
                # 尝试直接加载音频
                try:
                    audio_data, sr = librosa.load(decoded_path, sr=16000)
                    print(f"音频加载成功: 采样率={sr}, 长度={len(audio_data)}")
                except Exception as load_error:
                    print(f"直接加载失败: {load_error}")
                    
                    # 如果直接加载失败，尝试使用soundfile
                    try:
                        import soundfile as sf
                        print("尝试使用soundfile加载音频...")
                        audio_data, sr = sf.read(decoded_path)
                        if len(audio_data.shape) > 1:
                            audio_data = audio_data[:, 0]  # 取第一个声道
                        # 重采样到16kHz
                        if sr != 16000:
                            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                        print(f"soundfile加载成功: 采样率={sr}, 长度={len(audio_data)}")
                    except Exception as sf_error:
                        print(f"soundfile加载也失败: {sf_error}")
                        raise RuntimeError(f"无法加载音频文件: {load_error}")
                
                # 检查音频数据
                if len(audio_data) == 0:
                    raise ValueError("音频数据为空")
                
                # 音频数据预处理
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # 音频归一化
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                print(f"音频预处理完成: 数据类型={audio_data.dtype}, 范围=[{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
                
            except Exception as e:
                print(f"音频加载失败: {e}")
                return jsonify({"error": f"音频加载失败: {str(e)}"}), 500

            # 调用音素级评分函数
            try:
                print("开始调用音素级发音评分函数...")
                print(f"音频数据: 长度={len(audio_data)}, 类型={audio_data.dtype}")
                print(f"参考文本: '{reference_text}'")
                
                result = score_pronunciation_detailed(audio_data, reference_text)
                print(f"音素级评分完成")
                
                # 处理结果
                if hasattr(result, 'overall_score'):  # DetailedPronunciationResult对象
                    response_data = {
                        "overall_score": f"{result.overall_score:.1f}",
                        "phoneme_scores": [
                            {
                                "phoneme": ps.phoneme,
                                "start_time": ps.start_time,
                                "end_time": ps.end_time,
                                "score": ps.score,
                                "confidence": ps.confidence,
                                "quality": ps.quality,
                                "issues": ps.issues
                            } for ps in result.phoneme_scores
                        ],
                        "pronunciation_issues": result.pronunciation_issues,
                        "improvement_suggestions": result.improvement_suggestions,
                        "duration_analysis": result.duration_analysis,
                        "pitch_analysis": result.pitch_analysis,
                        "detailed": True
                    }
                elif isinstance(result, dict):  # 简化结果字典
                    response_data = {
                        "overall_score": f"{result['overall_score']:.1f}",
                        "phoneme_scores": result.get('phoneme_scores', []),
                        "pronunciation_issues": result.get('pronunciation_issues', []),
                        "improvement_suggestions": result.get('improvement_suggestions', []),
                        "detailed": result.get('detailed_available', False)
                    }
                else:  # 简单数值结果（向后兼容）
                    response_data = {
                        "overall_score": f"{result:.1f}",
                        "phoneme_scores": [],
                        "pronunciation_issues": [],
                        "improvement_suggestions": ["继续练习以提高发音准确度"],
                        "detailed": False
                    }
                
                return jsonify(response_data)
                
            except Exception as e:
                print(f"音素级评分函数调用失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 提供具体的错误信息
                error_msg = str(e)
                if "发音评分失败" in error_msg:
                    return jsonify({"error": error_msg}), 500
                else:
                    return jsonify({"error": f"音素级评分计算失败: {error_msg}"}), 500
                
        finally:
            # 清理临时文件
            try:
                if os.path.exists(audio_path) and not KEEP_UPLOADS:
                    os.remove(audio_path)
                    print(f"临时文件已清理: {audio_path}")
                # 同时清理可能的转码产物
                base, _ = os.path.splitext(audio_path)
                converted = f"{base}_16k.wav"
                if os.path.exists(converted) and not KEEP_UPLOADS:
                    os.remove(converted)
                    print(f"转换文件已清理: {converted}")
            except Exception as cleanup_error:
                print(f"清理临时文件时出错: {str(cleanup_error)}")
                
    except Exception as e:
        print(f"音素级发音评分接口错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"音素级评分过程中出错: {str(e)}"}), 500
#简化发音评分接口（备选方案）
@app.route('/api/score-pronunciation-simple', methods=['POST'])
def score_pronunciation_simple_api():
    """简化版本的发音评分，不依赖复杂的语音识别模型"""
    try:
        print("=== 开始处理简化发音评分请求 ===")
        
        # 获取请求参数
        reference_text = request.form.get('reference_text')
        audio_file = request.files.get('audio_file')

        print(f"参考文本: {reference_text}")
        print(f"音频文件: {audio_file.filename if audio_file else 'None'}")

        # 参数验证
        if not reference_text:
            print("错误: 缺少参考文本")
            return jsonify({'error': '缺少参考文本'}), 400
        if not audio_file:
            print("错误: 缺少音频文件")
            return jsonify({'error': '缺少音频文件'}), 400

        # 生成唯一的临时文件路径
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        import os
        _, ext = os.path.splitext(audio_file.filename)
        ext = (ext or '.webm').lower()
        audio_path = os.path.join(AUDIO_UPLOAD_DIR, f"temp_audio_{unique_id}{ext}")
        
        try:
            # 保存音频文件
            audio_file.save(audio_path)
            file_size = os.path.getsize(audio_path)
            print(f"音频文件已保存: {audio_path}, 大小: {file_size} 字节")

            # 检查文件是否成功保存
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print("错误: 音频文件保存失败或为空文件")
                return jsonify({"error": "音频文件保存失败或为空文件"}), 500

            # 简化的评分逻辑：基于音频文件大小和时长进行模拟评分
            try:
                import librosa
                print("正在分析音频文件...")
                audio_data, sr = librosa.load(audio_path, sr=16000)
                duration = len(audio_data) / sr
                print(f"音频时长: {duration:.2f}秒")
                
                # 简单的评分算法：基于音频时长和参考文本长度的匹配度
                expected_duration = len(reference_text.split()) * 0.5  # 假设每个单词0.5秒
                duration_score = max(0, 100 - abs(duration - expected_duration) * 20)
                
                # 添加一些随机性，让每次评分略有不同
                import random
                random.seed(hash(reference_text) % 1000)  # 基于文本的固定随机种子
                random_adjustment = random.uniform(-5, 5)
                
                final_score = max(0, min(100, duration_score + random_adjustment))
                print(f"简化评分完成: {final_score:.1f}")
                
                return jsonify({"score": f"{final_score:.1f}"})
                
            except Exception as e:
                print(f"音频分析失败: {e}")
                # 如果音频分析失败，返回一个基于文本长度的模拟分数
                text_length = len(reference_text)
                if text_length < 20:
                    base_score = 85
                elif text_length < 50:
                    base_score = 80
                else:
                    base_score = 75
                
                import random
                random.seed(hash(reference_text) % 1000)
                final_score = max(0, min(100, base_score + random.uniform(-10, 10)))
                print(f"使用备选评分: {final_score:.1f}")
                
                return jsonify({"score": f"{final_score:.1f}"})
                
        finally:
            # 清理临时文件
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    print(f"临时文件已清理: {audio_path}")
            except Exception as cleanup_error:
                print(f"清理临时文件时出错: {str(cleanup_error)}")
                
    except Exception as e:
        print(f"简化发音评分接口错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"评分过程中出错: {str(e)}"}), 500
#随机中文句子接口
@app.route('/api/random-chinese-sentence', methods=['GET'])
def get_random_chinese_sentence():
    tsv_file = os.path.join("data", "常用英语口语.txt")
    chinese_sentence = shuijizhongwen(tsv_file)
    if not chinese_sentence:
        return jsonify({"error": "No sentences found"}), 404
    return jsonify({"sentence": chinese_sentence})
#语法检测接口
@app.route('/api/check-grammar', methods=['POST'])
def check_grammar_api():
    try:
        # 获取请求参数（支持仅文本，音频可选）
        translated_text = request.form.get("translated_text")
        audio_file = request.files.get("audio_file")

        # 参数验证
        if not translated_text or not translated_text.strip():
            return jsonify({"error": "缺少翻译文本"}), 400

        transcribed_text = ""
        # 如提供音频，则尝试处理音频
        if audio_file and audio_file.filename:
            import uuid, os
            unique_id = uuid.uuid4().hex[:8]
            _, ext = os.path.splitext(audio_file.filename)
            ext = (ext or '.wav').lower()
            audio_path = os.path.join(AUDIO_UPLOAD_DIR, f"temp_recording_{unique_id}{ext}")

            try:
                audio_file.save(audio_path)

                # 如果为空文件，则忽略音频，继续仅基于文本做检查
                if (not os.path.exists(audio_path)) or os.path.getsize(audio_path) == 0:
                    audio_path = None

                if audio_path:
                    transcribed_text = transcribe_audio(audio_path)
            finally:
                try:
                    if audio_path and os.path.exists(audio_path):
                        os.remove(audio_path)
                except Exception as cleanup_error:
                    print(f"清理临时文件时出错: {str(cleanup_error)}")

        # 以用户文本为准进行语法分析
        analysis_result = analyze_grammar(translated_text)

        # 构建返回结果
        if analysis_result.get("status") == "success":
            result = {"status": "success", "message": "✅ 英文语法正确!"}
        else:
            result = {
                "status": "error",
                "error_count": analysis_result.get('error_count', 0),
                "errors": analysis_result.get('errors', [])
            }

        return jsonify({
            "translated_text": translated_text,
            "transcribed_text": transcribed_text,
            "result": result
        })
    except Exception as e:
        print(f"语法检测接口错误: {str(e)}")
        return jsonify({"error": f"语法检测过程中出错: {str(e)}"}), 500
@app.route('/api/custom-exercise', methods=['POST'])
def custom_exercise():
    data = request.json or {}
    file_path = data.get('file_path')
    input_text = data.get('text')  # 可选，文本内容（按行）
    mode = data.get('mode')  # 'speech' 或 'grammar'

    records = []
    try:
        if input_text and isinstance(input_text, str):
            lines = [line.strip() for line in input_text.splitlines() if line.strip()]
            records = [{"sentence": line} for line in lines]
        elif file_path:
            records = load_custom_data(file_path)
        else:
            records = []
    except Exception as e:
        return jsonify({'error': f'读取自定义数据失败: {str(e)}'})

    # 若无自定义数据，按模式回退到默认数据集
    if not records:
        if mode == 'speech':
            try:
                tsv_file = os.path.join("data", "common_voice", "validated.tsv")
                records = load_sentences_and_paths(tsv_file)
            except Exception:
                records = []
        elif mode == 'grammar':
            try:
                # 使用常用英语口语作为中文题面来源
                chinese = shuijizhongwen(os.path.join("data", "常用英语口语.txt"))
                if chinese:
                    records = [{"chinese": chinese}]
            except Exception:
                records = []
        # 仍无可用数据
        if not records:
            return jsonify({'error': '未提供有效的自定义数据！'})

    if mode == 'speech':
        random_record = get_random_custom_sentence(records)
        reference_text = random_record.get("sentence") or random_record.get("text") or ""
        if not reference_text:
            return jsonify({'error': '自定义数据中未找到可朗读文本！'})
        # 仅返回题目，录音与评分由前端完成并调用 /api/score-pronunciation
        return jsonify({
            'reference_text': reference_text
        })

    elif mode == 'grammar':
        random_record = get_random_custom_sentence(records)
        chinese_sentence = random_record.get("chinese") or random_record.get("sentence") or random_record.get("text") or ""
        if not chinese_sentence:
            return jsonify({'error': '自定义数据中未找到中文句子！'})
        # 仅返回题目，前端提交文本到 /api/check-grammar
        return jsonify({
            'chinese_sentence': chinese_sentence
        })

    else:
        return jsonify({'error': '无效选项！'})

if __name__ == '__main__':
    # 为了避免音素级评分时的自动重载问题，在生产环境中关闭调试模式
    app.run(debug=False, host='0.0.0.0', port=5000)