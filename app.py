from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from services.whisper_service import WhisperService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Whisper STT API",
    description="Whisper 기반 음성-텍스트 변환 서비스",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper_service = WhisperService()


@app.on_event("startup")
async def startup_event():
    logger.info("Whisper 모델 로딩 ")
    whisper_service.load_model()
    logger.info("Whisper 모델 로드 완료")


@app.get("/")
async def root():
    return {
        "message": "Whisper STT API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "transcribe": "POST /transcribe"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": whisper_service.is_model_loaded(),
        "service": "whisper-stt"
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        if not audio.filename:
            raise HTTPException(status_code=400, detail="파일 이름이 없습니다")

        file_extension = audio.filename.split('.')[-1].lower()
        allowed_extensions = ['mp3', 'wav', 'm4a', 'flac', 'ogg']

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 파일 형식입니다. 허용: {', '.join(allowed_extensions)}"
            )


        audio_data = await audio.read()

        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="빈 파일입니다")

        logger.info(f"음성 파일 수신: {audio.filename} ({len(audio_data)} bytes)")

        transcription = whisper_service.transcribe(audio_data, file_extension)

        logger.info(f"변환 완료: {transcription[:50]}...")

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "text": transcription,
                "filename": audio.filename
            }
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"변환 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"음성 변환 실패: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True 
    )
