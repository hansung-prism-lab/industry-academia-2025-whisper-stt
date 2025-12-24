import os
import sys
import tempfile
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data import _read_audio_any
from src.config import WhisperConfig, default_config
from src.exceptions import (
    ModelNotLoadedError,
    ModelLoadError,
    TranscriptionError,
    FFmpegNotFoundError,
    FFmpegConversionError,
    AudioConversionError
)

logger = logging.getLogger(__name__)


class WhisperService:


    ALLOWED_FORMATS = ["wav", "mp3", "m4a", "flac", "ogg"]
    FFMPEG_TIMEOUT = 30

    def __init__(
        self,
        config: Optional[WhisperConfig] = None,
        model_path: Optional[str] = None
    ):
        self.config = config or default_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[WhisperForConditionalGeneration] = None
        self.processor: Optional[WhisperProcessor] = None

        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = self.config.model.get_model_path()

        logger.info(f"WhisperService 초기화 완료 (device: {self.device})")
        logger.info(f"모델 경로: {self.model_path}")

    def is_model_loaded(self) -> bool:
        """모델 로드 여부 확인"""
        return self.model is not None and self.processor is not None

    def load_model(self) -> None:
        try:
            logger.info(f"모델 로딩 중: {self.model_path}")
            logger.info(f"사용 디바이스: {self.device}")

            self.processor = WhisperProcessor.from_pretrained(
                self.model_path,
                language=self.config.audio.language,
                task=self.config.audio.task
            )

            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_path
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info("모델 로드 완료")

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise ModelLoadError(
                message="Whisper 모델 로드 실패",
                path=self.model_path,
                cause=e
            )

    def _check_ffmpeg(self) -> None:
        """FFmpeg 설치 확인"""
        if shutil.which("ffmpeg") is None:
            raise FFmpegNotFoundError()

    def _convert_to_wav(self, input_path: str, output_path: str) -> None:

        self._check_ffmpeg()

        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', str(self.config.audio.sample_rate),
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            '-y',
            output_path
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.FFMPEG_TIMEOUT
            )

            if result.returncode != 0:
                stderr = result.stderr.decode() if result.stderr else "Unknown error"
                raise FFmpegConversionError(
                    message="FFmpeg 변환 실패",
                    stderr=stderr
                )

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise FFmpegConversionError(message="FFmpeg 출력 파일이 비어있습니다")

        except subprocess.TimeoutExpired:
            raise FFmpegConversionError(message="FFmpeg 변환 타임아웃")

    def transcribe(self, audio_data: bytes, file_extension: str = "mp3") -> str:

        if not self.is_model_loaded():
            raise ModelNotLoadedError()

        tmp_input_path = None
        tmp_wav_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=f".{file_extension}",
                delete=False
            ) as tmp_file:
                tmp_file.write(audio_data)
                tmp_input_path = tmp_file.name

            # WAV로 변환
            tmp_wav_fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_wav_fd)

            logger.info(f"오디오 변환 시작: {file_extension} -> WAV")
            self._convert_to_wav(tmp_input_path, tmp_wav_path)
            logger.info("오디오 변환 완료, Whisper 처리 시작")

            audio_array, sampling_rate = _read_audio_any(tmp_wav_path)

            input_features = self.processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_features
            input_features = input_features.to(self.device)

            # 추론
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    language=self.config.model.language,
                    task=self.config.model.task
                )

            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            return transcription.strip()

        except (ModelNotLoadedError, FFmpegNotFoundError, FFmpegConversionError):
            raise
        except Exception as e:
            logger.error(f"음성 변환 중 오류: {e}")
            raise TranscriptionError(message="음성 인식 실패", cause=e)

        finally:
            if tmp_input_path and os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)
            if tmp_wav_path and os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)

    def transcribe_file(self, file_path: str) -> str:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {file_path}")

        with open(file_path, 'rb') as f:
            audio_data = f.read()

        file_extension = Path(file_path).suffix[1:]
        return self.transcribe(audio_data, file_extension)
