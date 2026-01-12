#!/usr/bin/env python3
"""
AI Necklace OpenAI - Raspberry Pi 5 リアルタイム音声AIアシスタント（OpenAI版）

OpenAI Realtime APIを使用したリアルタイム双方向音声対話システム。
Gmail、アラーム、カメラ、音声メッセージ機能を統合。

機能:
- リアルタイム音声対話（低レイテンシ）
- Gmail連携（メール確認・返信・送信）
- アラーム機能（時刻指定で音声通知）
- カメラ機能（Gemini Visionで画像認識）
- 写真付きメール送信
- 音声メッセージ（Firebase経由でスマホとやり取り）
"""

import os
import sys
import json
import base64
import asyncio
import threading
import signal
import time
import re
import subprocess
import io
import wave
import tempfile
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import numpy as np
import pyaudio
import websockets

try:
    import alsaaudio
    ALSA_AVAILABLE = True
except ImportError:
    ALSA_AVAILABLE = False
    print("警告: alsaaudioが見つかりません。PyAudioで出力します。")

# Gemini Vision API (camera_capture用)
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Gmail API
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False
    print("警告: Google APIライブラリが見つかりません。Gmail機能は無効です。")

# Firebase Voice Messenger
try:
    from firebase_voice import FirebaseVoiceMessenger
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("警告: firebase_voiceモジュールが見つかりません。音声メッセージ機能は無効です。")

# GPIOライブラリ
try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("警告: gpiozeroが使用できません。ボタン操作は無効です。")

# systemdで実行時にprint出力をリアルタイムで表示するため
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 環境変数の読み込み（~/.ai-necklace/.env から）
env_path = os.path.expanduser("~/.ai-necklace/.env")
load_dotenv(env_path)

# =====================================================
# 会話ログ設定
# =====================================================
import logging
from logging.handlers import RotatingFileHandler

# ログディレクトリ作成
LOG_DIR = os.path.expanduser("~/.ai-necklace/logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 会話ログ専用のlogger
conversation_logger = logging.getLogger("conversation")
conversation_logger.setLevel(logging.INFO)

# ファイルハンドラ（10MB x 5ファイルでローテーション）
log_file = os.path.join(LOG_DIR, "conversation.log")
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
conversation_logger.addHandler(file_handler)

# コンソールにも出力
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
conversation_logger.addHandler(console_handler)

def log_conversation(role: str, content: str, extra: str = None):
    """会話ログを記録"""
    if extra:
        conversation_logger.info(f"[{role}] {content} ({extra})")
    else:
        conversation_logger.info(f"[{role}] {content}")

# Gmail APIスコープ
GMAIL_SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.modify'
]

# 設定
CONFIG = {
    # OpenAI Realtime API設定
    "model": "gpt-4o-realtime-preview",
    "voice": "alloy",  # OpenAI voices: alloy, echo, shimmer, fable, onyx, nova

    # オーディオ設定 (OpenAI Realtime API仕様)
    "send_sample_rate": 24000,    # OpenAI入力: 24kHz
    "receive_sample_rate": 24000,  # OpenAI出力: 24kHz
    "input_sample_rate": 48000,    # マイク入力: 48kHz
    "output_sample_rate": 48000,   # スピーカー出力: 48kHz
    "channels": 1,                 # モノラル
    "chunk_size": 1024,

    # デバイス設定（PyAudioで自動検出）
    "input_device_index": None,
    "output_device_index": None,

    # GPIO設定
    "button_pin": 5,
    "use_button": True,

    # Gmail設定
    "gmail_credentials_path": os.path.expanduser("~/.ai-necklace/credentials.json"),
    "gmail_token_path": os.path.expanduser("~/.ai-necklace/token.json"),

    # アラーム設定
    "alarm_file_path": os.path.expanduser("~/.ai-necklace/alarms.json"),

    # ライフログ設定
    "lifelog_dir": os.path.expanduser("~/lifelog"),
    "lifelog_interval": 60,  # 1分（秒）

    # システムプロンプト
    "instructions": """あなたは日本語で会話する自律型AIアシスタントです。
ユーザーの意図を推測し、必要なツールを自分で判断して使います。

【言語設定 - 最重要】
- ユーザーは必ず日本語で話します
- 音声認識結果が韓国語、中国語、アラビア語、ロシア語、英語、その他の言語として表示された場合、それは誤認識です
- 誤認識された場合は「すみません、聞き取れませんでした。もう一度お願いします」と日本語で返答してください
- 誤認識された場合は絶対にツールを呼び出さないでください
- 日本語として正しく認識された場合のみ、以下のルールに従ってください

【自律的なツール選択】
あなたは「執事」のように、ユーザーが明示的に指示しなくても文脈から意図を推測してツールを使います。

■ 視覚が必要な質問 → camera_capture を使用
  例：
  - 「この答えは何？」→ カメラで目の前を撮影し、画像から「答え」を推測して回答
  - 「これ何？」→ カメラで撮影し、目の前の物体を説明
  - 「どう思う？」（目の前のものについて）→ カメラで撮影し、意見を述べる
  - 「読んで」→ カメラで撮影し、テキストを読み上げる
  - 「何が見える？」→ カメラで撮影し、視界の内容を説明
  - 「色は？」「サイズは？」→ カメラで確認して回答
  - 「どっちがいい？」（選択肢を見せながら）→ カメラで見て意見を述べる

■ メール関連の話題 → gmail系ツールを使用
  例：
  - 「新着メールある？」「メール来てる？」→ gmail_list
  - 「○○さんからのメール読んで」→ gmail_list → gmail_read
  - 「今のメールに返信して」→ gmail_reply
  - 「○○さんにメール送って」→ gmail_send
  - 「写真を○○に送って」→ gmail_send_photo

■ 時間・リマインダー関連 → alarm系ツールを使用
  例：
  - 「7時に起こして」「30分後に教えて」→ alarm_set
  - 「アラーム確認」→ alarm_list
  - 「アラーム消して」→ alarm_delete

■ スマホへの連絡 → voice_send / voice_send_photo を使用
  例：
  - 「スマホにメッセージ」「妻に連絡」→ voice_send
  - 「この写真をスマホに送って」→ voice_send_photo

■ ライフログ関連 → lifelog系ツールを使用
  例：
  - 「記録開始」「ライフログ開始」→ lifelog_start
  - 「記録停止」→ lifelog_stop
  - 「今日何枚撮った？」→ lifelog_status

【推論の流れ】
1. ユーザーの発話を分析
2. 「この質問に答えるには何が必要か？」を考える
3. 必要なツールがあれば使用
4. ツールの結果を元に自然な回答を生成

【ツールを使わない場合】
- 純粋な会話・雑談：「おはよう」「元気？」「何ができるの？」
- 一般知識の質問：「東京タワーの高さは？」「1+1は？」
- ユーザーが「ツールを使わないで」と明示した場合

【注意事項】
- 迷ったときは、まず会話で確認してもOK（「目の前の物について聞いていますか？」）
- ただし、文脈から明らかな場合は確認せずにツールを使う
- 「～して」という依頼には積極的にツールを使う
""",
}

# グローバル変数
running = True
button = None
is_recording = False
openai_client = None

# Gmail関連
gmail_service = None
last_email_list = []

# アラーム関連
alarms = []
alarm_next_id = 1

# Firebase関連
firebase_messenger = None
voice_message_mode = False  # 音声メッセージ録音モード
voice_message_mode_timestamp = None  # 音声メッセージモード開始時刻
VOICE_MESSAGE_MODE_TIMEOUT = 60  # 60秒でタイムアウト
voice_message_buffer = []   # 録音バッファ


def check_and_reset_voice_message_mode():
    """voice_message_modeのタイムアウトをチェックし、必要ならリセット"""
    global voice_message_mode, voice_message_mode_timestamp

    if voice_message_mode and voice_message_mode_timestamp:
        elapsed = time.time() - voice_message_mode_timestamp
        if elapsed > VOICE_MESSAGE_MODE_TIMEOUT:
            print(f"voice_message_mode タイムアウト ({elapsed:.1f}秒経過) - リセット")
            voice_message_mode = False
            voice_message_mode_timestamp = None
            return False  # タイムアウトでリセットされた
    return voice_message_mode


def reset_voice_message_mode():
    """voice_message_modeを強制リセット（再接続時などに使用）"""
    global voice_message_mode, voice_message_mode_timestamp

    if voice_message_mode:
        print("voice_message_mode を強制リセット")
    voice_message_mode = False
    voice_message_mode_timestamp = None

# ライフログ関連
lifelog_enabled = False
lifelog_thread = None
lifelog_photo_count = 0  # 今日の撮影枚数

# カメラ排他制御用ロック
camera_lock = threading.Lock()

# グローバルオーディオハンドラ（スマホからの音声再生用）
global_audio_handler = None


def signal_handler(sig, frame):
    """Ctrl+C で終了"""
    global running
    print("\n終了します...")
    running = False


# ==================== ユーティリティ ====================

def find_audio_device(p, device_type="input"):
    """オーディオデバイスを自動検出"""
    # 入力デバイス用の名前リスト（USB機器を優先、defaultは最後の手段）
    input_target_names = ["usbmic", "USB PnP Sound", "USB Audio", "USB PnP Audio"]
    # 出力デバイス用の名前リスト（UACDemo=USBスピーカーを優先）
    output_target_names = ["usbspk", "UACDemo", "USB Audio", "USB PnP Audio"]
    target_names = input_target_names if device_type == "input" else output_target_names

    # デバッグ: 全デバイスを表示
    print(f"=== オーディオデバイス一覧 ({device_type}) ===")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get("name", "")
        in_ch = info.get("maxInputChannels", 0)
        out_ch = info.get("maxOutputChannels", 0)
        print(f"  [{i}] {name} (入力:{in_ch}ch, 出力:{out_ch}ch)")

    # USBデバイスを探す
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get("name", "")

        if device_type == "input" and info.get("maxInputChannels", 0) > 0:
            for target in target_names:
                if target.lower() in name.lower():
                    print(f"入力デバイス検出: [{i}] {name}")
                    return i
        elif device_type == "output" and info.get("maxOutputChannels", 0) > 0:
            for target in target_names:
                if target.lower() in name.lower():
                    print(f"出力デバイス検出: [{i}] {name}")
                    return i

    # 見つからなければデフォルト
    print(f"{device_type}デバイスが見つかりません。デフォルトを使用。")
    return None


def resample_audio(audio_data, from_rate, to_rate):
    """オーディオデータをリサンプリング"""
    if from_rate == to_rate:
        return audio_data

    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

    # 整数倍のダウンサンプリング（高品質）
    if from_rate > to_rate and from_rate % to_rate == 0:
        factor = from_rate // to_rate
        # 整数倍でダウンサンプリング（平均を取る方が品質が良い）
        trim_length = (len(audio_array) // factor) * factor
        trimmed = audio_array[:trim_length]
        resampled = trimmed.reshape(-1, factor).mean(axis=1)
    else:
        # 一般的なリサンプリング
        original_length = len(audio_array)
        new_length = int(original_length * to_rate / from_rate)
        indices = np.linspace(0, original_length - 1, new_length)
        resampled = np.interp(indices, np.arange(original_length), audio_array)

    return resampled.astype(np.int16).tobytes()


def mono_to_stereo(audio_data):
    """モノラルをステレオに変換"""
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    stereo = np.column_stack((audio_array, audio_array))
    return stereo.flatten().tobytes()


# ==================== 効果音生成 ====================

def generate_beep_wav(frequency=800, duration=0.15):
    """ビープ音のWAVデータを生成"""
    try:
        sample_rate = 48000
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, False)

        # シンプルなサイン波
        wave_data = np.sin(2 * np.pi * frequency * t)

        # フェードイン/アウト
        fade_samples = int(samples * 0.1)
        wave_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
        wave_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        # 16bit PCM
        wave_data = (wave_data * 32767 * 0.5).astype(np.int16)

        # WAVファイルとして出力
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(wave_data.tobytes())
        return buffer.getvalue()
    except Exception as e:
        print(f"ビープ音生成エラー: {e}")
        return None


def generate_double_beep_wav(frequency1=600, frequency2=800, duration1=0.1, duration2=0.15):
    """2音のビープ音のWAVデータを生成（起動音）"""
    try:
        sample_rate = 48000
        t1 = np.linspace(0, duration1, int(sample_rate * duration1), False)
        wave1 = np.sin(2 * np.pi * frequency1 * t1)
        wave1 = (wave1 * 32767 * 0.4).astype(np.int16)

        gap = np.zeros(int(sample_rate * 0.1), dtype=np.int16)

        t2 = np.linspace(0, duration2, int(sample_rate * duration2), False)
        wave2 = np.sin(2 * np.pi * frequency2 * t2)
        wave2 = (wave2 * 32767 * 0.5).astype(np.int16)

        wave_data = np.concatenate([wave1, gap, wave2])

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(wave_data.tobytes())
        return buffer.getvalue()
    except Exception as e:
        print(f"2音ビープ生成エラー: {e}")
        return None


def generate_alarm_sound_wav(duration=3.0):
    """アラーム音のWAVデータを生成"""
    try:
        sample_rate = 48000
        total_samples = int(sample_rate * duration)
        wave_data = np.zeros(total_samples, dtype=np.float32)

        beep_duration = 0.2
        gap_duration = 0.15
        cycle_duration = beep_duration + gap_duration
        num_cycles = int(duration / cycle_duration)

        for i in range(num_cycles):
            start = int(i * cycle_duration * sample_rate)
            end = int(start + beep_duration * sample_rate)
            if end > total_samples:
                end = total_samples

            t = np.linspace(0, beep_duration, end - start, False)
            freq = 880 if i % 2 == 0 else 1100
            wave_data[start:end] = np.sin(2 * np.pi * freq * t) * 0.6

        wave_data = (wave_data * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(wave_data.tobytes())
        return buffer.getvalue()
    except Exception as e:
        print(f"アラーム音生成エラー: {e}")
        return None


# ==================== Gmail機能 ====================

def init_gmail():
    """Gmail APIを初期化"""
    global gmail_service

    if not GMAIL_AVAILABLE:
        return False

    creds = None
    token_path = CONFIG["gmail_token_path"]
    credentials_path = CONFIG["gmail_credentials_path"]

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"トークン更新エラー: {e}")
                return False
        else:
            if not os.path.exists(credentials_path):
                print(f"Gmail認証ファイルが見つかりません: {credentials_path}")
                return False
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, GMAIL_SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                print(f"Gmail認証エラー: {e}")
                return False

        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    try:
        gmail_service = build('gmail', 'v1', credentials=creds)
        return True
    except Exception as e:
        print(f"Gmail初期化エラー: {e}")
        return False


def gmail_list_func(query="is:unread", max_results=5):
    """メール一覧を取得"""
    global last_email_list

    if not gmail_service:
        return "Gmailが初期化されていません"

    try:
        results = gmail_service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()

        messages = results.get('messages', [])
        if not messages:
            return "該当するメールはありません"

        email_list = []
        for i, msg in enumerate(messages, 1):
            msg_data = gmail_service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='metadata',
                metadataHeaders=['Subject', 'From', 'Date']
            ).execute()

            headers = {h['name']: h['value'] for h in msg_data.get('payload', {}).get('headers', [])}
            email_list.append({
                'id': msg['id'],
                'index': i,
                'from': headers.get('From', '不明'),
                'subject': headers.get('Subject', '(件名なし)'),
                'date': headers.get('Date', '')
            })

        last_email_list = email_list

        result_text = f"メールが{len(email_list)}件あります:\n"
        for email in email_list:
            result_text += f"{email['index']}. {email['from']}: {email['subject']}\n"

        return result_text

    except Exception as e:
        return f"メール取得エラー: {e}"


def gmail_read_func(message_id):
    """メール本文を取得"""
    if not gmail_service:
        return "Gmailが初期化されていません"

    # 番号で指定された場合
    if isinstance(message_id, int) or (isinstance(message_id, str) and message_id.isdigit()):
        index = int(message_id)
        if last_email_list and 1 <= index <= len(last_email_list):
            message_id = last_email_list[index - 1]['id']
        else:
            return f"メール番号{index}が見つかりません。先に「メールを確認して」と言ってください。"

    try:
        msg = gmail_service.users().messages().get(
            userId='me',
            id=message_id,
            format='full'
        ).execute()

        payload = msg.get('payload', {})
        headers = {h['name']: h['value'] for h in payload.get('headers', [])}

        # 本文を取得
        body = ""
        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain':
                    data = part.get('body', {}).get('data', '')
                    if data:
                        body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                        break
        else:
            data = payload.get('body', {}).get('data', '')
            if data:
                body = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

        # 長すぎる場合は切り詰める
        if len(body) > 500:
            body = body[:500] + "...(省略)"

        return f"件名: {headers.get('Subject', '(なし)')}\n差出人: {headers.get('From', '不明')}\n本文:\n{body}"

    except Exception as e:
        return f"メール読み取りエラー: {e}"


def gmail_send_func(to, subject, body):
    """新規メールを送信"""
    if not gmail_service:
        return "Gmailが初期化されていません"

    try:
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        gmail_service.users().messages().send(
            userId='me',
            body={'raw': raw}
        ).execute()

        return f"メールを送信しました: {to}"

    except Exception as e:
        return f"メール送信エラー: {e}"


def gmail_reply_func(message_id, body):
    """メールに返信"""
    if not gmail_service:
        return "Gmailが初期化されていません"

    # 番号で指定された場合
    if isinstance(message_id, int) or (isinstance(message_id, str) and message_id.isdigit()):
        index = int(message_id)
        if last_email_list and 1 <= index <= len(last_email_list):
            message_id = last_email_list[index - 1]['id']
        else:
            return f"メール番号{index}が見つかりません"

    try:
        original = gmail_service.users().messages().get(
            userId='me',
            id=message_id,
            format='metadata',
            metadataHeaders=['Subject', 'From', 'Message-ID']
        ).execute()

        headers = {h['name']: h['value'] for h in original.get('payload', {}).get('headers', [])}
        to = headers.get('From', '')
        subject = headers.get('Subject', '')
        if not subject.startswith('Re:'):
            subject = 'Re: ' + subject

        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        message['In-Reply-To'] = headers.get('Message-ID', '')
        message['References'] = headers.get('Message-ID', '')

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        gmail_service.users().messages().send(
            userId='me',
            body={'raw': raw, 'threadId': original.get('threadId')}
        ).execute()

        return f"返信しました: {to}"

    except Exception as e:
        return f"返信エラー: {e}"


def gmail_send_photo_func(to, subject, body):
    """写真付きメールを送信（カメラで撮影して送信）"""
    if not gmail_service:
        return "Gmailが初期化されていません"

    with camera_lock:
        try:
            # カメラで撮影
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name

            result = subprocess.run(
                ['libcamera-still', '-o', tmp_path, '-t', '1000', '--nopreview', '-n'],
                capture_output=True,
                timeout=10
            )

            if result.returncode != 0:
                return "写真撮影に失敗しました"

            with open(tmp_path, 'rb') as f:
                image_data = f.read()

            os.unlink(tmp_path)

            # メール作成
            message = MIMEMultipart()
            message['to'] = to
            message['subject'] = subject

            message.attach(MIMEText(body))

            part = MIMEBase('image', 'jpeg')
            part.set_payload(image_data)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment', filename='photo.jpg')
            message.attach(part)

            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            gmail_service.users().messages().send(
                userId='me',
                body={'raw': raw}
            ).execute()

            return f"写真付きメールを送信しました: {to}"

        except Exception as e:
            return f"写真付きメール送信エラー: {e}"


# ==================== アラーム機能 ====================

def load_alarms():
    """アラームをファイルから読み込み"""
    global alarms, alarm_next_id
    alarm_file = CONFIG["alarm_file_path"]

    if os.path.exists(alarm_file):
        try:
            with open(alarm_file, 'r') as f:
                data = json.load(f)
                alarms = data.get('alarms', [])
                alarm_next_id = data.get('next_id', 1)
        except Exception as e:
            print(f"アラーム読み込みエラー: {e}")
            alarms = []
            alarm_next_id = 1


def save_alarms():
    """アラームをファイルに保存"""
    alarm_file = CONFIG["alarm_file_path"]
    try:
        with open(alarm_file, 'w') as f:
            json.dump({'alarms': alarms, 'next_id': alarm_next_id}, f)
    except Exception as e:
        print(f"アラーム保存エラー: {e}")


def alarm_set_func(time_str, label="", message=""):
    """アラームを設定"""
    global alarm_next_id

    # 時刻をパース（HH:MM または H:MM 形式）
    time_match = re.match(r'(\d{1,2}):(\d{2})', time_str)
    if not time_match:
        # 「7時」「7時半」などの形式
        hour_match = re.match(r'(\d{1,2})時(半)?', time_str)
        if hour_match:
            hour = int(hour_match.group(1))
            minute = 30 if hour_match.group(2) else 0
        else:
            return f"時刻の形式が不正です: {time_str}"
    else:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))

    alarm = {
        'id': alarm_next_id,
        'hour': hour,
        'minute': minute,
        'label': label,
        'message': message or f"{hour}時{minute}分のアラームです",
        'active': True
    }
    alarms.append(alarm)
    alarm_next_id += 1
    save_alarms()

    return f"アラームをセットしました: {hour}時{minute:02d}分"


def alarm_list_func():
    """アラーム一覧を取得"""
    if not alarms:
        return "アラームはセットされていません"

    result = "アラーム一覧:\n"
    for alarm in alarms:
        if alarm['active']:
            result += f"- ID{alarm['id']}: {alarm['hour']}時{alarm['minute']:02d}分 {alarm['label']}\n"

    return result


def alarm_delete_func(alarm_id=None):
    """アラームを削除"""
    global alarms

    if alarm_id is None:
        # 全削除
        alarms = []
        save_alarms()
        return "全てのアラームを削除しました"

    for i, alarm in enumerate(alarms):
        if alarm['id'] == alarm_id:
            del alarms[i]
            save_alarms()
            return f"アラームID{alarm_id}を削除しました"

    return f"アラームID{alarm_id}が見つかりません"


def check_alarms():
    """アラームをチェックして通知"""
    global global_audio_handler, openai_client

    now = datetime.now()
    for alarm in alarms[:]:
        if alarm['active'] and alarm['hour'] == now.hour and alarm['minute'] == now.minute:
            alarm['active'] = False
            save_alarms()

            print(f"アラーム発動: {alarm['message']}")

            # アラーム音を再生
            if global_audio_handler:
                alarm_sound = generate_alarm_sound_wav()
                if alarm_sound:
                    global_audio_handler.play_audio_buffer(alarm_sound)

            # AIにアラームを通知（テキストメッセージとして送信）
            if openai_client and openai_client.is_connected:
                asyncio.run_coroutine_threadsafe(
                    openai_client.send_text_message(
                        f"[アラーム通知] {alarm['message']} ユーザーに知らせてください。"
                    ),
                    openai_client.loop
                )


def alarm_check_thread():
    """アラーム監視スレッド"""
    global running
    last_check_minute = -1

    while running:
        now = datetime.now()
        if now.minute != last_check_minute:
            last_check_minute = now.minute
            check_alarms()
        time.sleep(1)


def start_alarm_thread():
    """アラーム監視スレッドを開始"""
    thread = threading.Thread(target=alarm_check_thread, daemon=True)
    thread.start()
    print("アラーム監視スレッド開始")


# ==================== カメラ機能 ====================

def camera_capture_func(prompt="この画像に何が写っていますか？"):
    """カメラで撮影してGemini Vision APIで分析"""
    with camera_lock:
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name

            result = subprocess.run(
                ['libcamera-still', '-o', tmp_path, '-t', '1000', '--nopreview', '-n'],
                capture_output=True,
                timeout=10
            )

            if result.returncode != 0:
                return "カメラ撮影に失敗しました"

            with open(tmp_path, 'rb') as f:
                image_data = f.read()

            os.unlink(tmp_path)

            # Gemini Vision APIで分析
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                return "Gemini APIキーが設定されていません"

            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Part.from_bytes(
                        data=image_data,
                        mime_type="image/jpeg"
                    ),
                    prompt
                ]
            )

            return response.text

        except subprocess.TimeoutExpired:
            return "カメラタイムアウト"
        except Exception as e:
            return f"カメラエラー: {e}"


# ==================== 音声メッセージ機能 ====================

def init_firebase():
    """Firebase Voice Messengerを初期化"""
    global firebase_messenger

    if not FIREBASE_AVAILABLE:
        return False

    try:
        firebase_messenger = FirebaseVoiceMessenger()
        firebase_messenger.start_polling()
        return True
    except Exception as e:
        print(f"Firebase初期化エラー: {e}")
        return False


def voice_send_func():
    """音声メッセージモードを開始"""
    global voice_message_mode, voice_message_mode_timestamp

    if not firebase_messenger:
        return "Firebaseが初期化されていません"

    voice_message_mode = True
    voice_message_mode_timestamp = time.time()
    print(f"音声メッセージモード開始 (タイムアウト: {VOICE_MESSAGE_MODE_TIMEOUT}秒)")
    return "音声メッセージを録音します。ボタンを押して話してください。"


def send_recorded_voice_message():
    """録音した音声をFirebaseに送信"""
    global voice_message_mode, voice_message_mode_timestamp, voice_message_buffer, global_audio_handler

    if not firebase_messenger:
        return False

    try:
        audio_handler = global_audio_handler
        if not audio_handler:
            print("オーディオハンドラがありません")
            return False

        # 録音
        voice_message_buffer = []
        if not audio_handler.start_input_stream():
            print("録音開始失敗")
            return False

        print("音声メッセージ録音中...")

        # ボタンが押されている間録音
        while button and button.is_pressed and running:
            chunk = audio_handler.read_audio_chunk()
            if chunk:
                voice_message_buffer.append(chunk)
            time.sleep(0.02)

        audio_handler.stop_input_stream()

        if not voice_message_buffer:
            print("録音データがありません")
            return False

        # WAVファイル作成
        audio_data = b''.join(voice_message_buffer)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(CONFIG["send_sample_rate"])
            wf.writeframes(audio_data)

        wav_data = wav_buffer.getvalue()

        # Firebaseに送信
        result = firebase_messenger.send_audio(wav_data, "audio/wav")
        if result:
            print("音声メッセージを送信しました")
            return True
        else:
            print("音声メッセージ送信失敗")
            return False

    except Exception as e:
        print(f"音声メッセージエラー: {e}")
        return False
    finally:
        voice_message_mode = False
        voice_message_mode_timestamp = None
        voice_message_buffer = []


def voice_send_photo_func():
    """カメラで撮影してFirebaseに送信"""
    if not firebase_messenger:
        return "Firebaseが初期化されていません"

    with camera_lock:
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name

            result = subprocess.run(
                ['libcamera-still', '-o', tmp_path, '-t', '1000', '--nopreview', '-n'],
                capture_output=True,
                timeout=10
            )

            if result.returncode != 0:
                return "写真撮影に失敗しました"

            with open(tmp_path, 'rb') as f:
                image_data = f.read()

            os.unlink(tmp_path)

            result = firebase_messenger.send_image(image_data, "image/jpeg")
            if result:
                return "写真をスマホに送信しました"
            else:
                return "写真送信に失敗しました"

        except Exception as e:
            return f"写真送信エラー: {e}"


# ==================== ライフログ機能 ====================

def lifelog_capture():
    """ライフログ用に写真を撮影してFirebaseにアップロード"""
    global lifelog_photo_count

    if not firebase_messenger:
        return False

    with camera_lock:
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name

            result = subprocess.run(
                ['libcamera-still', '-o', tmp_path, '-t', '500', '--nopreview', '-n'],
                capture_output=True,
                timeout=10
            )

            if result.returncode != 0:
                return False

            with open(tmp_path, 'rb') as f:
                image_data = f.read()

            os.unlink(tmp_path)

            # Firebase Storageにアップロード
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result = firebase_messenger.upload_lifelog(image_data, timestamp)

            if result:
                lifelog_photo_count += 1
                return True
            return False

        except Exception as e:
            print(f"ライフログ撮影エラー: {e}")
            return False


def lifelog_thread_func():
    """ライフログ撮影スレッド"""
    global running, lifelog_enabled, lifelog_photo_count

    last_date = datetime.now().date()

    while running:
        if lifelog_enabled:
            # 日付が変わったらカウントをリセット
            current_date = datetime.now().date()
            if current_date != last_date:
                last_date = current_date
                lifelog_photo_count = 0
                print(f"日付変更: ライフログカウントをリセット")

            lifelog_capture()

        time.sleep(CONFIG["lifelog_interval"])


def start_lifelog_thread():
    """ライフログスレッドを開始"""
    global lifelog_thread
    lifelog_thread = threading.Thread(target=lifelog_thread_func, daemon=True)
    lifelog_thread.start()
    print("ライフログスレッド開始")


def lifelog_start_func():
    """ライフログを開始"""
    global lifelog_enabled
    lifelog_enabled = True
    return f"ライフログを開始しました（{CONFIG['lifelog_interval']}秒間隔）"


def lifelog_stop_func():
    """ライフログを停止"""
    global lifelog_enabled
    lifelog_enabled = False
    return "ライフログを停止しました"


def lifelog_status_func():
    """ライフログステータスを取得"""
    global lifelog_photo_count, lifelog_enabled

    status = "有効" if lifelog_enabled else "停止中"
    return f"ライフログ: {status}、今日の撮影枚数: {lifelog_photo_count}枚"


# ==================== ツール実行 ====================

def execute_tool(tool_name, arguments):
    """ツールを実行"""
    log_conversation("TOOL", f"{tool_name}", f"args: {arguments}")
    print(f"ツール実行: {tool_name} - {arguments}")

    if tool_name == "gmail_list":
        return gmail_list_func(
            query=arguments.get("query", "is:unread"),
            max_results=arguments.get("max_results", 5)
        )
    elif tool_name == "gmail_read":
        return gmail_read_func(arguments.get("message_id"))
    elif tool_name == "gmail_send":
        return gmail_send_func(
            arguments.get("to"),
            arguments.get("subject"),
            arguments.get("body")
        )
    elif tool_name == "gmail_reply":
        return gmail_reply_func(
            arguments.get("message_id"),
            arguments.get("body")
        )
    elif tool_name == "gmail_send_photo":
        return gmail_send_photo_func(
            arguments.get("to"),
            arguments.get("subject"),
            arguments.get("body", "写真を送ります")
        )
    elif tool_name == "alarm_set":
        return alarm_set_func(
            arguments.get("time"),
            arguments.get("label", ""),
            arguments.get("message", "")
        )
    elif tool_name == "alarm_list":
        return alarm_list_func()
    elif tool_name == "alarm_delete":
        return alarm_delete_func(arguments.get("alarm_id"))
    elif tool_name == "camera_capture":
        return camera_capture_func(arguments.get("prompt", "この画像に何が写っていますか？"))
    elif tool_name == "voice_send":
        return voice_send_func()
    elif tool_name == "voice_send_photo":
        return voice_send_photo_func()
    elif tool_name == "lifelog_start":
        return lifelog_start_func()
    elif tool_name == "lifelog_stop":
        return lifelog_stop_func()
    elif tool_name == "lifelog_status":
        return lifelog_status_func()
    else:
        return f"不明なツール: {tool_name}"


# ==================== OpenAI ツール定義 ====================

def get_openai_tools():
    """OpenAI Realtime API用のツール定義を取得"""
    tools = [
        {
            "type": "function",
            "name": "gmail_list",
            "description": """メール一覧を取得。以下の状況で自動的に使用：
- 「メールある？」「新着メールは？」「メール来てる？」
- 「○○さんからメール来てる？」（from:で検索）
- メールについて話題が出たとき""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索クエリ（例: is:unread, from:xxx@gmail.com）"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "取得件数（デフォルト5）"
                    }
                },
                "required": []
            }
        },
        {
            "type": "function",
            "name": "gmail_read",
            "description": "メールの本文を読む。gmail_listで取得したメール番号を指定。",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "メールID または 番号（1, 2, 3...）"
                    }
                },
                "required": ["message_id"]
            }
        },
        {
            "type": "function",
            "name": "gmail_send",
            "description": "新規メールを送信",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "宛先メールアドレス"
                    },
                    "subject": {
                        "type": "string",
                        "description": "件名"
                    },
                    "body": {
                        "type": "string",
                        "description": "本文"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        },
        {
            "type": "function",
            "name": "gmail_reply",
            "description": "メールに返信。gmail_listで取得したメール番号を指定。",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "返信先メールID または 番号"
                    },
                    "body": {
                        "type": "string",
                        "description": "返信本文"
                    }
                },
                "required": ["message_id", "body"]
            }
        },
        {
            "type": "function",
            "name": "gmail_send_photo",
            "description": "カメラで撮影して写真付きメールを送信",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "宛先メールアドレス"
                    },
                    "subject": {
                        "type": "string",
                        "description": "件名"
                    },
                    "body": {
                        "type": "string",
                        "description": "本文"
                    }
                },
                "required": ["to", "subject"]
            }
        },
        {
            "type": "function",
            "name": "alarm_set",
            "description": "アラームを設定。「7時に起こして」「30分後に教えて」など。",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "時刻（例: 7:00, 7時, 7時半）"
                    },
                    "label": {
                        "type": "string",
                        "description": "アラームのラベル"
                    },
                    "message": {
                        "type": "string",
                        "description": "アラーム時に読み上げるメッセージ"
                    }
                },
                "required": ["time"]
            }
        },
        {
            "type": "function",
            "name": "alarm_list",
            "description": "設定されているアラーム一覧を取得",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "type": "function",
            "name": "alarm_delete",
            "description": "アラームを削除",
            "parameters": {
                "type": "object",
                "properties": {
                    "alarm_id": {
                        "type": "integer",
                        "description": "削除するアラームのID（省略で全削除）"
                    }
                },
                "required": []
            }
        },
        {
            "type": "function",
            "name": "camera_capture",
            "description": """カメラで目の前を撮影して分析。【自律的に使用】以下の状況で積極的に使用：
■ 視覚情報が必要な質問すべて：
- 「この答えは何？」→ 目の前の問題を撮影し答えを計算
- 「これ何？」「何が見える？」→ 目の前を撮影して説明
- 「読んで」→ テキストを撮影して読み上げ
- 「どう思う？」（目の前のものについて）→ 撮影して意見
- 「色は？」「サイズは？」→ 撮影して確認""",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "画像に対する質問（省略可）"
                    }
                },
                "required": []
            }
        },
        {
            "type": "function",
            "name": "voice_send",
            "description": "スマホに音声メッセージを送信。ボタン押下で録音開始。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "type": "function",
            "name": "voice_send_photo",
            "description": "カメラで撮影してスマホに写真を送信",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "type": "function",
            "name": "lifelog_start",
            "description": "ライフログ自動撮影を開始（1分間隔）",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "type": "function",
            "name": "lifelog_stop",
            "description": "ライフログ自動撮影を停止",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "type": "function",
            "name": "lifelog_status",
            "description": "ライフログの状態と今日の撮影枚数を確認",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]
    return tools


# ==================== OpenAI Audio Handler ====================

class OpenAIAudioHandler:
    """OpenAI Realtime API用音声処理ハンドラ"""

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.is_recording = False
        self.is_playing = False

    def start_input_stream(self):
        """マイク入力開始（48kHz）"""
        if self.input_stream:
            return True

        try:
            input_device = CONFIG["input_device_index"]
            if input_device is None:
                input_device = find_audio_device(self.audio, "input")

            self.input_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=CONFIG["channels"],
                rate=CONFIG["input_sample_rate"],
                input=True,
                input_device_index=input_device,
                frames_per_buffer=CONFIG["chunk_size"]
            )
            self.is_recording = True
            print("マイク入力開始")
            return True
        except Exception as e:
            print(f"マイク入力開始エラー: {e}")
            return False

    def stop_input_stream(self):
        """マイク入力停止"""
        if self.input_stream:
            print("マイク入力停止")
            self.is_recording = False
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
            except Exception:
                pass
            self.input_stream = None

    def read_audio_chunk(self):
        """音声チャンクを読み取り、24kHzにリサンプリング"""
        if self.input_stream and self.is_recording:
            try:
                data = self.input_stream.read(CONFIG["chunk_size"], exception_on_overflow=False)
                # 48kHz -> 24kHz ダウンサンプリング
                resampled = resample_audio(data, CONFIG["input_sample_rate"], CONFIG["send_sample_rate"])
                return resampled
            except Exception as e:
                print(f"音声読み取りエラー: {e}")
        return None

    def start_output_stream(self):
        """スピーカー出力開始（48kHz）"""
        if self.output_stream:
            return

        try:
            output_device = CONFIG["output_device_index"]
            if output_device is None:
                output_device = find_audio_device(self.audio, "output")

            self.output_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=CONFIG["channels"],
                rate=CONFIG["output_sample_rate"],
                output=True,
                output_device_index=output_device,
                frames_per_buffer=CONFIG["chunk_size"] * 2
            )
            self.is_playing = True
            print("スピーカー出力開始")
        except Exception as e:
            print(f"スピーカー出力開始エラー: {e}")

    def stop_output_stream(self):
        """スピーカー出力停止"""
        if self.output_stream:
            print("スピーカー出力停止")
            self.is_playing = False
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
            except Exception:
                pass
            self.output_stream = None

    def play_audio_chunk(self, audio_data_base64):
        """Base64エンコードされたPCM（24kHz）を48kHzにリサンプリングして再生"""
        if self.output_stream and self.is_playing:
            try:
                # Base64デコード
                audio_data = base64.b64decode(audio_data_base64)
                # 24kHz -> 48kHz リサンプリング
                resampled = resample_audio(audio_data, CONFIG["receive_sample_rate"], CONFIG["output_sample_rate"])
                self.output_stream.write(resampled)
            except Exception as e:
                print(f"音声再生エラー: {e}")

    def play_audio_buffer(self, audio_data):
        """完全なWAVバッファを再生（起動音、アラーム用）"""
        if not self.output_stream or not self.is_playing:
            return

        try:
            wav_buffer = io.BytesIO(audio_data)
            with wave.open(wav_buffer, 'rb') as wf:
                original_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())

                # リサンプリング（必要な場合）
                if original_rate != CONFIG["output_sample_rate"]:
                    audio_np = np.frombuffer(frames, dtype=np.int16)
                    ratio = CONFIG["output_sample_rate"] / original_rate
                    new_length = int(len(audio_np) * ratio)
                    indices = np.linspace(0, len(audio_np) - 1, new_length).astype(int)
                    resampled = audio_np[indices]
                    frames = resampled.astype(np.int16).tobytes()

                # チャンク単位で書き込み
                chunk_size = 4096
                for i in range(0, len(frames), chunk_size):
                    if not self.is_playing:
                        break
                    self.output_stream.write(frames[i:i+chunk_size])
        except Exception as e:
            print(f"バッファ再生エラー: {e}")

    def cleanup(self):
        """リソース解放"""
        self.stop_input_stream()
        self.stop_output_stream()
        if self.audio:
            self.audio.terminate()


# ==================== OpenAI Realtime Client ====================

class OpenAIRealtimeClient:
    """OpenAI Realtime APIクライアント"""

    WEBSOCKET_URL = "wss://api.openai.com/v1/realtime"
    MODEL = "gpt-4o-realtime-preview"
    MAX_RECONNECT_ATTEMPTS = 5
    RECONNECT_DELAY_BASE = 2
    SESSION_RESET_TIMEOUT = 30

    def __init__(self, audio_handler: OpenAIAudioHandler):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY が設定されていません")

        self.audio_handler = audio_handler
        self.websocket = None
        self.is_connected = False
        self.is_responding = False
        self.pending_tool_calls = {}
        self.loop = None
        self.needs_reconnect = False
        self.reconnect_count = 0
        self.needs_session_reset = False
        self.last_response_time = None

    def get_session_config(self):
        """session.updateイベント用の設定を取得"""
        return {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": CONFIG["instructions"],
                "voice": CONFIG["voice"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": None,  # VAD無効化（プッシュトゥトーク）
                "tools": get_openai_tools(),
                "tool_choice": "auto"
            }
        }

    async def connect(self):
        """WebSocket接続を確立"""
        print(f"OpenAI Realtime APIに接続中... ({self.MODEL})")

        try:
            url = f"{self.WEBSOCKET_URL}?model={self.MODEL}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }

            self.websocket = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            self.is_connected = True
            self.loop = asyncio.get_event_loop()

            # session.createdを待機
            message = await self.websocket.recv()
            data = json.loads(message)
            if data.get("type") == "session.created":
                print("OpenAI Realtime API接続完了")
                # セッション設定を送信
                await self.send_session_update()
            else:
                print(f"予期しないメッセージ: {data.get('type')}")

        except Exception as e:
            print(f"接続エラー: {e}")
            self.is_connected = False
            raise

    async def disconnect(self):
        """WebSocket接続を切断"""
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
            self.is_connected = False
            print("OpenAI Realtime API切断")

    async def reconnect(self):
        """指数バックオフで再接続"""
        global running

        self.reconnect_count += 1
        if self.reconnect_count > self.MAX_RECONNECT_ATTEMPTS:
            print(f"再接続回数が上限({self.MAX_RECONNECT_ATTEMPTS}回)に達しました。プログラムを終了します。")
            running = False
            return False

        delay = min(self.RECONNECT_DELAY_BASE ** self.reconnect_count, 60)
        print(f"{delay}秒後に再接続を試みます... (試行 {self.reconnect_count}/{self.MAX_RECONNECT_ATTEMPTS})")
        await asyncio.sleep(delay)

        await self.disconnect()
        self.needs_reconnect = False
        self.pending_tool_calls = {}

        if not voice_message_mode:
            reset_voice_message_mode()

        try:
            await self.connect()
            print("再接続成功!")
            return True
        except Exception as e:
            print(f"再接続失敗: {e}")
            self.needs_reconnect = True
            return False

    async def reset_session(self):
        """セッションをリセットして新しい会話を開始"""
        print("セッションリセット中...")

        await self.disconnect()
        self.pending_tool_calls = {}

        if not voice_message_mode:
            reset_voice_message_mode()

        try:
            await self.connect()
            print("セッションリセット完了 - 新しい会話を開始")
            return True
        except Exception as e:
            print(f"セッションリセット失敗: {e}")
            self.needs_reconnect = True
            return False

    async def send_session_update(self):
        """session.updateイベントを送信"""
        config = self.get_session_config()
        await self.websocket.send(json.dumps(config))
        print("セッション設定送信完了")

    async def send_audio_chunk(self, audio_data):
        """音声チャンクをBase64エンコードして送信"""
        if not self.is_connected or not self.websocket:
            return

        try:
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            event = {
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            }
            await self.websocket.send(json.dumps(event))
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\n音声送信エラー: {e}")
            self.needs_reconnect = True

    async def commit_audio_buffer(self):
        """音声バッファをコミット（録音終了時）"""
        if not self.is_connected or not self.websocket:
            return

        try:
            event = {"type": "input_audio_buffer.commit"}
            await self.websocket.send(json.dumps(event))
            print("audio_buffer.commit送信")
        except Exception as e:
            print(f"コミットエラー: {e}")

    async def clear_audio_buffer(self):
        """音声バッファをクリア"""
        if not self.is_connected or not self.websocket:
            return

        try:
            event = {"type": "input_audio_buffer.clear"}
            await self.websocket.send(json.dumps(event))
        except Exception as e:
            print(f"クリアエラー: {e}")

    async def create_response(self):
        """応答生成をトリガー"""
        if not self.is_connected or not self.websocket:
            return

        try:
            event = {"type": "response.create"}
            await self.websocket.send(json.dumps(event))
            print("response.create送信")
            self.is_responding = True
        except Exception as e:
            print(f"応答生成エラー: {e}")

    async def cancel_response(self):
        """進行中の応答をキャンセル"""
        if not self.is_connected or not self.websocket:
            return

        try:
            event = {"type": "response.cancel"}
            await self.websocket.send(json.dumps(event))
            self.is_responding = False
        except Exception as e:
            print(f"キャンセルエラー: {e}")

    async def send_text_message(self, text):
        """テキストメッセージを送信（アラーム通知用）"""
        if not self.is_connected or not self.websocket:
            return

        try:
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": text
                        }
                    ]
                }
            }
            await self.websocket.send(json.dumps(event))
            await self.create_response()
        except Exception as e:
            print(f"テキスト送信エラー: {e}")

    async def send_tool_response(self, call_id, output):
        """ツール実行結果を送信"""
        if not self.is_connected or not self.websocket:
            return

        try:
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({"result": output}) if isinstance(output, str) else json.dumps(output)
                }
            }
            await self.websocket.send(json.dumps(event))
            print("ツール結果送信完了")
        except Exception as e:
            print(f"ツール結果送信エラー: {e}")

    async def receive_messages(self):
        """サーバーからのメッセージを受信"""
        global running

        print("受信ループ開始")
        while running and self.is_connected:
            try:
                print("receive()呼び出し待機中...")
                message = await self.websocket.recv()
                data = json.loads(message)
                await self.handle_message(data)
                self.reconnect_count = 0  # 正常受信でリセット
            except websockets.exceptions.ConnectionClosed as e:
                print(f"接続が閉じられました: {e}")
                self.is_connected = False
                self.needs_reconnect = True
                break
            except Exception as e:
                print(f"受信エラー: {e}")
                self.is_connected = False
                self.needs_reconnect = True
                break

    async def handle_message(self, message):
        """受信メッセージを処理"""
        event_type = message.get("type")

        # セッション更新完了
        if event_type == "session.updated":
            print("セッション設定完了")

        # 音声デルタ（音声データ受信）
        elif event_type == "response.audio.delta":
            audio_base64 = message.get("delta")
            if audio_base64:
                self.audio_handler.play_audio_chunk(audio_base64)

        # 音声トランスクリプト（AIの発話テキスト）
        elif event_type == "response.audio_transcript.done":
            text = message.get("transcript")
            if text:
                log_conversation("AI", text)

        # 入力音声トランスクリプト完了
        elif event_type == "conversation.item.input_audio_transcription.completed":
            text = message.get("transcript")
            if text:
                log_conversation("USER", text)

        # ツール呼び出し
        elif event_type == "response.function_call_arguments.done":
            await self.handle_tool_call(message)

        # 応答完了
        elif event_type == "response.done":
            self.is_responding = False
            self.last_response_time = time.time()
            log_conversation("SYSTEM", "--- 応答完了 ---")

        # エラー
        elif event_type == "error":
            error = message.get("error", {})
            print(f"APIエラー: {error.get('message', error)}")

    async def handle_tool_call(self, message):
        """ツール呼び出しを処理"""
        call_id = message.get("call_id")
        tool_name = message.get("name")
        arguments_str = message.get("arguments", "{}")

        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            arguments = {}

        # 長時間かかるツールは別スレッドで実行
        if tool_name in ["voice_send_photo", "camera_capture", "gmail_send_photo"]:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: execute_tool(tool_name, arguments))
        else:
            result = execute_tool(tool_name, arguments)

        # ツール結果を送信
        await self.send_tool_response(call_id, result)
        # 応答生成をトリガー
        await self.create_response()


# ==================== 音声入力ループ ====================

async def audio_input_loop(client: OpenAIRealtimeClient, audio_handler: OpenAIAudioHandler):
    """音声入力ループ"""
    global running, button, is_recording, voice_message_mode

    while running:
        if CONFIG["use_button"] and button:
            if button.is_pressed:
                if not is_recording:
                    # セッションリセット中または未接続の場合は待機
                    if client.needs_session_reset or not client.is_connected:
                        await asyncio.sleep(0.1)
                        continue

                    # 音声メッセージモード確認（タイムアウト付き）
                    current_voice_mode = check_and_reset_voice_message_mode()

                    if current_voice_mode:
                        # 音声メッセージ録音モード
                        log_conversation("SYSTEM", "=== 音声メッセージ録音開始 ===")
                        is_recording = True
                        loop = asyncio.get_event_loop()
                        success = await loop.run_in_executor(None, send_recorded_voice_message)
                        is_recording = False
                        # 音声メッセージ送信完了後にセッションリセット
                        client.needs_session_reset = True
                        continue
                    else:
                        # 通常の音声入力モード
                        # ボタンが押されたのでセッションリセットタイマーをキャンセル
                        client.last_response_time = None
                        log_conversation("SYSTEM", "=== ボタン押下 - 録音開始 ===")

                        # 進行中の応答があればキャンセル
                        if client.is_responding:
                            await client.cancel_response()

                        # 音声バッファをクリア
                        await client.clear_audio_buffer()

                        if audio_handler.start_input_stream():
                            is_recording = True
                        else:
                            log_conversation("SYSTEM", "録音開始失敗")
                            continue

                # ボタン押下中は音声を送信し続ける
                chunk = audio_handler.read_audio_chunk()
                if chunk and len(chunk) > 0:
                    await client.send_audio_chunk(chunk)
            else:
                if is_recording:
                    is_recording = False
                    audio_handler.stop_input_stream()
                    log_conversation("SYSTEM", "=== ボタン離す - 録音停止 ===")

                    # 音声バッファをコミットして応答をトリガー
                    await client.commit_audio_buffer()
                    await client.create_response()

        await asyncio.sleep(0.01)


# ==================== メインループ ====================

async def main_async():
    """非同期メインループ（自動再接続対応）"""
    global running, openai_client, global_audio_handler

    # オーディオハンドラ初期化
    audio_handler = OpenAIAudioHandler()
    audio_handler.start_output_stream()
    global_audio_handler = audio_handler

    # OpenAIクライアント初期化
    client = OpenAIRealtimeClient(audio_handler)
    openai_client = client

    receive_task = None
    input_task = None
    first_start = True

    try:
        # バックグラウンドタスク開始
        start_alarm_thread()
        start_lifelog_thread()

        while running:
            # タイムアウトによるセッションリセットチェック（30秒間操作がなければリセット）
            if client.last_response_time and client.is_connected:
                elapsed = time.time() - client.last_response_time
                if elapsed >= client.SESSION_RESET_TIMEOUT:
                    log_conversation("SYSTEM", f"--- {client.SESSION_RESET_TIMEOUT}秒間操作なし - セッションリセット ---")
                    client.needs_session_reset = True
                    client.last_response_time = None

            # セッションリセットが必要な場合
            if client.needs_session_reset and client.is_connected:
                client.needs_session_reset = False
                print("セッションリセット実行中...")

                # 受信タスクをキャンセル
                if receive_task and not receive_task.done():
                    receive_task.cancel()
                    try:
                        await receive_task
                    except asyncio.CancelledError:
                        pass

                # セッションをリセット
                await client.reset_session()

                # 新しい受信タスクを開始
                if client.is_connected:
                    receive_task = asyncio.create_task(client.receive_messages())
                print("セッションリセット完了 - 新しい会話の準備完了")

            # 接続されていない場合は接続を試みる
            if not client.is_connected:
                if client.needs_reconnect:
                    # 再接続
                    success = await client.reconnect()
                    if not success:
                        continue
                else:
                    # 初回接続
                    try:
                        await client.connect()
                    except Exception as e:
                        print(f"接続エラー: {e}")
                        print("5秒後に再試行します...")
                        await asyncio.sleep(5)
                        continue

                # タスクを開始/再開
                if receive_task is None or receive_task.done():
                    receive_task = asyncio.create_task(client.receive_messages())
                if input_task is None or input_task.done():
                    input_task = asyncio.create_task(audio_input_loop(client, audio_handler))

                if first_start:
                    first_start = False

                    # 起動情報表示
                    print("=" * 50)
                    print("AI Necklace OpenAI Realtime 起動（全機能版）")
                    print("=" * 50)
                    print(f"Gmail: {'有効' if gmail_service else '無効'}")
                    print(f"Firebase: {'有効' if firebase_messenger else '無効'}")
                    print(f"アラーム: {len(alarms)}件")
                    print(f"カメラ: 有効")
                    print(f"ライフログ: {'実行中' if lifelog_enabled else '待機中'}（{CONFIG['lifelog_interval']}秒間隔）")
                    if CONFIG["use_button"]:
                        print(f"操作: GPIO{CONFIG['button_pin']}のボタンを押している間話す")
                    print("=" * 50)
                    print("コマンド例:")
                    print("  - 「メールを確認して」")
                    print("  - 「写真を撮って」「何が見える？」")
                    print("  - 「7時にアラームをセット」")
                    print("  - 「スマホにメッセージを送って」")
                    print("  - 「ライフログ開始」「ライフログ停止」")
                    print("=" * 50)
                    print("--- ボタンを押して話しかけてください ---")

                    # 起動音を再生
                    startup_sound = generate_double_beep_wav()
                    if startup_sound:
                        print("音声バッファ再生中...")
                        audio_handler.play_audio_buffer(startup_sound)

                    print("起動完了")
                else:
                    # 再接続時は短い通知音
                    print("再接続完了 - 会話を再開できます")

            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"メインループエラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("終了処理中...")
        await client.disconnect()
        audio_handler.cleanup()
        print("終了しました")


def main():
    """エントリーポイント"""
    global button, running

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("初期化中...")

    # Gmail初期化
    if init_gmail():
        print("Gmail: 有効")
    else:
        print("Gmail: 無効")

    # Firebase初期化
    if init_firebase():
        print("Firebase Voice Messenger: 有効")
    else:
        print("Firebase Voice Messenger: 無効")

    # アラーム読み込み
    load_alarms()
    print(f"アラーム: {len(alarms)}件読み込み")

    # ボタン初期化
    if CONFIG["use_button"] and GPIO_AVAILABLE:
        try:
            button = Button(CONFIG["button_pin"], pull_up=True, bounce_time=0.1)
            print(f"ボタン: GPIO{CONFIG['button_pin']}")
        except Exception as e:
            print(f"ボタン初期化エラー: {e}")
            button = None
            CONFIG["use_button"] = False
    else:
        button = None
        if CONFIG["use_button"]:
            CONFIG["use_button"] = False

    # 非同期メインループ実行
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n終了します...")
    finally:
        running = False


if __name__ == "__main__":
    main()
