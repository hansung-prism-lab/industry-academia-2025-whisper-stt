from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class AudioConfig:
    """오디오 처리 설정"""
    sample_rate: int = 16000
    language: str = "Korean"
    task: str = "transcribe"


@dataclass
class ModelConfig:
    """모델 설정"""
    default_model: str = "openai/whisper-base"
    local_model_dir: str = "./output"
    language: str = "ko"
    task: str = "transcribe"

    def get_model_path(self) -> str:
        env_path = os.getenv("WHISPER_MODEL_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        if os.path.exists(self.local_model_dir):
            return self.local_model_dir

        return self.default_model


@dataclass
class TrainingConfig:
    """학습 설정"""
    output_dir: str = "output"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    num_train_epochs: int = 5
    gradient_checkpointing: bool = True
    fp16: bool = False
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 10
    save_steps: int = 100
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "cer"
    greater_is_better: bool = False
    weight_decay: float = 0.01
    generation_max_length: int = 225


@dataclass
class PreprocessingConfig:
    """전처리 설정"""
    checkpoint: str = "pyannote/voice-activity-detection"
    access_token: Optional[str] = None
    sample_rate: int = 16000
    hop_length: int = 512
    frame_length: int = 1024
    min_segment_duration: float = 0.2
    min_split_duration: float = 0.3
    max_split_duration: float = 3.0
    energy_percentile: int = 20
    energy_multiplier: float = 3.0
    low_energy_percentile: int = 10
    low_energy_multiplier: float = 1.5
    min_gap_duration: float = 0.5


@dataclass
class PathConfig:
    """경로 설정"""
    base_dir: str = field(default_factory=os.getcwd)
    data_dir: str = "data"
    audio_subdir: str = "audio"
    label_subdir: str = "label"
    output_dir: str = "output"

    @property
    def audio_dir(self) -> str:
        return os.path.join(self.base_dir, self.data_dir, self.audio_subdir)

    @property
    def label_dir(self) -> str:
        return os.path.join(self.base_dir, self.data_dir, self.label_subdir)

    @property
    def full_output_dir(self) -> str:
        return os.path.join(self.base_dir, self.output_dir)


@dataclass
class WhisperConfig:
    """Whisper STT 전체 설정"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_env(cls) -> "WhisperConfig":
        """환경변수에서 설정 로드"""
        config = cls()

        if os.getenv("WHISPER_SAMPLE_RATE"):
            config.audio.sample_rate = int(os.getenv("WHISPER_SAMPLE_RATE"))
        if os.getenv("WHISPER_MODEL_PATH"):
            config.model.local_model_dir = os.getenv("WHISPER_MODEL_PATH")
        if os.getenv("WHISPER_EPOCHS"):
            config.training.num_train_epochs = int(os.getenv("WHISPER_EPOCHS"))
        if os.getenv("WHISPER_BATCH_SIZE"):
            config.training.per_device_train_batch_size = int(os.getenv("WHISPER_BATCH_SIZE"))
        if os.getenv("WHISPER_LEARNING_RATE"):
            config.training.learning_rate = float(os.getenv("WHISPER_LEARNING_RATE"))

        return config


default_config = WhisperConfig()
