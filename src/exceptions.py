class WhisperException(Exception):
    pass


class AudioLoadError(WhisperException):
    """오디오 파일 로드 실패"""

    def __init__(self, message: str = "오디오 파일을 로드할 수 없습니다", path: str = None, cause: Exception = None):
        self.path = path
        self.cause = cause
        if path:
            message = f"{message}: {path}"
        if cause:
            message = f"{message} ({str(cause)})"
        super().__init__(message)


class AudioFormatError(WhisperException):
    """지원하지 않는 오디오 포맷"""

    def __init__(self, ext: str, allowed_formats: list = None):
        self.ext = ext
        self.allowed_formats = allowed_formats or ["wav", "mp3", "m4a", "flac", "ogg"]
        message = f"지원하지 않는 오디오 포맷: {ext}. 허용 포맷: {', '.join(self.allowed_formats)}"
        super().__init__(message)


class AudioConversionError(WhisperException):
    """오디오 변환 실패"""

    def __init__(self, message: str = "오디오 변환에 실패했습니다", cause: Exception = None):
        self.cause = cause
        if cause:
            message = f"{message}: {str(cause)}"
        super().__init__(message)


class ModelNotLoadedError(WhisperException):
    """모델이 로드되지 않음"""

    def __init__(self, message: str = "모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요"):
        super().__init__(message)


class ModelLoadError(WhisperException):
    """모델 로드 실패"""

    def __init__(self, message: str = "모델을 로드할 수 없습니다", path: str = None, cause: Exception = None):
        self.path = path
        self.cause = cause
        if path:
            message = f"{message}: {path}"
        if cause:
            message = f"{message} ({str(cause)})"
        super().__init__(message)


class TranscriptionError(WhisperException):
    """음성 인식 실패"""

    def __init__(self, message: str = "음성 인식 중 오류가 발생했습니다", cause: Exception = None):
        self.cause = cause
        if cause:
            message = f"{message}: {str(cause)}"
        super().__init__(message)


class FFmpegNotFoundError(WhisperException):
    """FFmpeg를 찾을 수 없음"""

    def __init__(self, message: str = "FFmpeg가 설치되어 있지 않습니다. FFmpeg를 설치해주세요"):
        super().__init__(message)


class FFmpegConversionError(WhisperException):
    """FFmpeg 변환 실패"""

    def __init__(self, message: str = "FFmpeg 변환에 실패했습니다", stderr: str = None):
        self.stderr = stderr
        if stderr:
            message = f"{message}: {stderr}"
        super().__init__(message)


class VADError(WhisperException):
    """Voice Activity Detection 오류"""

    def __init__(self, message: str = "음성 활동 감지 중 오류가 발생했습니다", cause: Exception = None):
        self.cause = cause
        if cause:
            message = f"{message}: {str(cause)}"
        super().__init__(message)


class SegmentationError(WhisperException):
    """세그먼트 처리 오류"""

    def __init__(self, message: str = "오디오 세그먼트 처리 중 오류가 발생했습니다", cause: Exception = None):
        self.cause = cause
        if cause:
            message = f"{message}: {str(cause)}"
        super().__init__(message)


class DatasetError(WhisperException):
    """데이터셋 처리 오류"""

    def __init__(self, message: str = "데이터셋 처리 중 오류가 발생했습니다", cause: Exception = None):
        self.cause = cause
        if cause:
            message = f"{message}: {str(cause)}"
        super().__init__(message)


class TrainingError(WhisperException):
    """학습 중 오류"""

    def __init__(self, message: str = "모델 학습 중 오류가 발생했습니다", cause: Exception = None):
        self.cause = cause
        if cause:
            message = f"{message}: {str(cause)}"
        super().__init__(message)
