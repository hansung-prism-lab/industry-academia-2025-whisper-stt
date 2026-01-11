# 경청 (Listening) - Whisper STT

<div align="center">


</div>

**경청**은 청각 및 음성 장애를 가진 사용자들을 위한 AI 기반 음성 인식(STT) 서비스입니다.
OpenAI Whisper 모델을 한국어에 fine-tuning하여 높은 정확도의 실시간 음성-텍스트 변환을 제공합니다.

## Preview
<img width="4959" height="7016" alt="preview" src="https://github.com/user-attachments/assets/38d82a6f-c280-4f17-a67e-775384586fa2" />

### Members

<table width="50%" align="center">
    <tr>
        <td align="center"><b>FE</b></td>
        <td align="center"><b>BE</b></td>
        <td align="center"><b>Whisper</b></td>
        <td align="center"><b>STT</b></td>
    </tr>
    <tr>
        <td align="center"><img src="https://github.com/user-attachments/assets/b95eea07-c69a-4bbf-9a8f-eccda41c410e" style="width:220px; object-fit:cover;" /></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/b72c8c11-6cd0-4569-aa9c-9caced8f6892" style="width:220px; object-fit:cover;" /></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/b02c39a0-532e-4f40-8943-9f3f197a8ce5" style="width:220px; object-fit:cover;" /></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/74c2f1a2-0cd1-4ed1-8dd1-1a5b44a7e43b" style="width:220px; object-fit:cover;" /></td>
    </tr>
    <tr>
        <td align="center"><b><a href="https://github.com/nyun-nye">윤예진</a></b></td>
        <td align="center"><b><a href="https://github.com/Lee-Han-Jun">이한준</a></b></td>
        <td align="center"><b><a href="https://github.com/hoya04">신정호</a></b></td>
        <td align="center"><b><a href="https://github.com/fhhdjsjs">최준희</a></b></td>
    </tr>
</table>
## Tech Stack

### Core Framework
- **FastAPI 0.118.2** - 고성능 비동기 웹 프레임워크
- **Python 3.10+** - 메인 개발 언어
- **Uvicorn** - ASGI 서버

### AI & Deep Learning
- **OpenAI Whisper** - 음성 인식 기반 모델
- **Transformers 4.57.0** - Hugging Face 트랜스포머
- **PyTorch 2.8.0** - 딥러닝 프레임워크
- **TorchAudio 2.8.0** - 오디오 처리

### Audio Processing
- **Librosa 0.11.0** - 오디오 분석 및 특징 추출
- **SoundFile 0.13.1** - 오디오 파일 I/O
- **Pyannote.audio 4.0.2** - 음성 활동 감지 (VAD)
- **FFmpeg** - 오디오 포맷 변환

### Data Processing
- **NumPy 2.3.3** - 수치 연산
- **Pandas 2.3.3** - 데이터 처리
- **Matplotlib 3.10.7** - 데이터 시각화

## Getting Started

### Prerequisites

- **Python 3.10+**
- **FFmpeg** (오디오 변환용)
- **CUDA** (선택사항, GPU 가속)

### Installation

```bash
# 저장소 클론
git clone https://github.com/YOUR_REPO/listening.git
cd listening/whisper-stt

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### FFmpeg 설치

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
- [FFmpeg 공식 사이트](https://ffmpeg.org/download.html)에서 다운로드
- 환경 변수 PATH에 추가

### Run

```bash
# 개발 서버 시작
python app.py

# 또는 uvicorn 직접 실행
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

서버가 실행되면 다음 주소에서 접근 가능:
- API: http://localhost:8001
- Health Check: http://localhost:8001/health
- API 문서: http://localhost:8001/docs

## Project Structure

```
whisper-stt/
├── app.py                       # FastAPI 메인 애플리케이션
├── requirements.txt             # Python 의존성
├── services/
│   ├── whisper_service.py       # Whisper STT 핵심 로직
│   ├── train.py                 # 모델 학습 스크립트
│   ├── Preprocessing.py         # 전처리 진입점
│   └── preprocessing/           # 전처리 모듈
│       ├── audio_io.py          # 오디오 파일 I/O
│       ├── vad.py               # 음성 활동 감지 (VAD)
│       ├── segment.py           # 오디오 세그멘테이션
│       ├── transcript.py        # 전사 텍스트 처리
│       ├── pipeline.py          # 전처리 파이프라인
│       ├── dataset.py           # 데이터셋 생성
│       └── config.py            # 전처리 설정
├── src/
│   ├── config.py                # 전역 설정 (AudioConfig, ModelConfig)
│   ├── model.py                 # Whisper 모델 초기화
│   ├── data.py                  # 데이터 로딩 및 전처리
│   ├── args.py                  # CLI 인자 파싱
│   ├── utils.py                 # 유틸리티 함수
│   ├── path.py                  # 경로 관리
│   └── exceptions.py            # 커스텀 예외 정의
├── data/                        # 학습 데이터 (미포함)
│   ├── audio/                   # 원본 오디오 파일
│   └── label/                   # 라벨 텍스트 파일
└── output/                      # Fine-tuned 모델 저장
```

## Key Features

### 1. 음성-텍스트 변환 (STT)

- **다중 오디오 포맷 지원**: MP3, WAV, M4A, FLAC, OGG
- **자동 포맷 변환**: FFmpeg 기반 16kHz WAV 변환
- **실시간 처리**: 비동기 FastAPI로 고성능 처리
- **한국어 특화**: Fine-tuned Whisper 모델 사용

### 2. Whisper 모델 Fine-tuning

- **전이 학습**: OpenAI Whisper-base를 한국어 데이터셋으로 fine-tuning
- **데이터 전처리 파이프라인**:
  - VAD (Voice Activity Detection)로 음성 구간 감지
  - 에너지 기반 세그멘테이션 (0.3~3.0초)
  - 자동 전사 텍스트 정렬
- **학습 최적화**: Gradient checkpointing, warmup, weight decay

### 3. RESTful API

- **POST /transcribe**: 음성 파일 업로드 및 텍스트 변환
- **GET /health**: 서비스 상태 및 모델 로드 확인
- **Swagger 문서**: `/docs`에서 대화형 API 테스트 가능

### 4. 에러 핸들링

- **커스텀 예외**:
  - `ModelNotLoadedError`: 모델 미로드 상태
  - `FFmpegNotFoundError`: FFmpeg 미설치
  - `TranscriptionError`: 음성 인식 실패
- **상세 로깅**: 각 단계별 처리 과정 추적

### 5. 설정 관리

- **환경변수 지원**: `WHISPER_MODEL_PATH`, `WHISPER_SAMPLE_RATE` 등
- **유연한 구성**: `src/config.py`에서 모든 하이퍼파라미터 관리
- **GPU 자동 감지**: CUDA 가용 시 자동 활성화

## API Usage

### 음성 파일 변환 예제

**cURL:**
```bash
curl -X POST "http://localhost:8001/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@your_audio_file.mp3"
```

**Python:**
```python
import requests

url = "http://localhost:8001/transcribe"
files = {"audio": open("your_audio_file.mp3", "rb")}
response = requests.post(url, files=files)

print(response.json())
# {'success': True, 'text': '변환된 텍스트', 'filename': 'your_audio_file.mp3'}
```

**JavaScript (Fetch):**
```javascript
const formData = new FormData();
formData.append('audio', audioFile);

fetch('http://localhost:8001/transcribe', {
  method: 'POST',
  body: formData
})
  .then(res => res.json())
  .then(data => console.log(data.text));
```

### Health Check

```bash
curl http://localhost:8001/health
# {'status': 'healthy', 'model_loaded': True, 'service': 'whisper-stt'}
```

## Model Training

Fine-tuning을 위한 학습 스크립트:

```bash
# 데이터 전처리
python services/Preprocessing.py

# 모델 학습
python services/train.py \
  --epochs 5 \
  --batch_size 2 \
  --learning_rate 1e-5 \
  --output_dir ./output
```

### 학습 데이터 구조

```
data/
├── audio/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── label/
    ├── sample1.txt
    ├── sample2.txt
    └── ...
```

각 오디오 파일에 대응하는 라벨 텍스트 파일이 필요합니다.

## Technical Highlights

### 1. FFmpeg 기반 오디오 정규화

모든 입력 오디오를 16kHz, 모노 채널, PCM 16-bit WAV로 변환:

```python
cmd = [
    'ffmpeg',
    '-i', input_path,
    '-ar', '16000',        # Sample rate
    '-ac', '1',            # Mono channel
    '-c:a', 'pcm_s16le',   # PCM 16-bit
    '-y',
    output_path
]
```

### 2. Whisper 모델 추론 파이프라인

```python
# 1. 오디오 로드
audio_array, sampling_rate = _read_audio_any(wav_path)

# 2. Feature 추출
input_features = processor(
    audio_array,
    sampling_rate=sampling_rate,
    return_tensors="pt"
).input_features

# 3. 추론
predicted_ids = model.generate(
    input_features,
    language="ko",
    task="transcribe"
)

# 4. 디코딩
transcription = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True
)[0]
```

### 3. VAD (Voice Activity Detection)

Pyannote.audio를 사용한 음성 구간 감지:

```python
# 음성/비음성 구간 분리
vad_segments = vad_pipeline(audio_path)

# 에너지 기반 세그멘테이션
segments = split_by_energy(
    audio,
    min_duration=0.3,
    max_duration=3.0
)
```

### 4. 임시 파일 관리

안전한 임시 파일 처리:

```python
try:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        tmp_file.write(audio_data)
        tmp_input_path = tmp_file.name

    # 처리...
finally:
    if tmp_input_path and os.path.exists(tmp_input_path):
        os.unlink(tmp_input_path)  # 자동 정리
```

## Performance


### 정확도 (한국어 Fine-tuned 모델)

- **CER (Character Error Rate)**: ~8.5%
- **WER (Word Error Rate)**: ~12.3%

*CPU 환경에서는 GPU 대비 3-5배 느림*

## License

이 프로젝트는 한성대학교 산학공동연구 프로젝트로 진행되었습니다.
