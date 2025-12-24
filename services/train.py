import glob
import os
import pandas as pd
import logging

from transformers import (
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer
)

from datasets import Dataset, DatasetDict
from datasets import Audio
import evaluate

from src.utils import load_txt
from src.data import prepare_dataset
from src.model import DataCollatorSpeechSeq2SeqWithPadding
from src.path import AUDIO_DIR, LABEL_DIR
from src.args import TRAINING_ARGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    logger.info("데이터 로드 중")

    audio_data_list = sorted(glob.glob(os.path.join(AUDIO_DIR, "*")))
    label_data_list = sorted(glob.glob(os.path.join(LABEL_DIR, "*")))

    logger.info(f"오디오 파일: {len(audio_data_list)}개")
    logger.info(f"라벨 파일: {len(label_data_list)}개")

    df = pd.DataFrame(columns=["audio", "label"])
    df["audio"] = audio_data_list
    df["label"] = [load_txt(i) for i in label_data_list]

    logger.info(f"데이터 로드 완료: {len(df)}개 샘플")
    logger.info(f"샘플 데이터:\n{df.head()}")

    return df


def prepare_datasets(df, feature_extractor, tokenizer):
    logger.info("데이터셋 전처리 중")

    ds = Dataset.from_dict(
        {
            "audio": [path for path in df["audio"]],
            "label": [label for label in df["label"]]
        }
    ).cast_column("audio", Audio(sampling_rate=16000, decode=False))

    logger.info("Train/Test/Valid 분할 중")
    train_testvalid = ds.train_test_split(test_size=0.2)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)

    datasets = DatasetDict({
        "train": train_testvalid["train"],
        "test": test_valid["test"],
        "valid": test_valid["train"]
    })

    logger.info(f"Train: {len(datasets['train'])}개")
    logger.info(f"Test: {len(datasets['test'])}개")
    logger.info(f"Valid: {len(datasets['valid'])}개")

    logger.info("Feature extraction 진행 중...")
    processed_datasets = datasets.map(
        prepare_dataset,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
        remove_columns=datasets["train"].column_names
    )

    logger.info("데이터셋 전처리 완료")
    return processed_datasets


def compute_metrics(pred, tokenizer, metric):
    """평가 메트릭 계산"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


def main():
    logger.info("Whisper Fine-tuning 시작")

    MODEL_CKPT = "openai/whisper-base"
    logger.info(f"모델: {MODEL_CKPT}")

    processor = WhisperProcessor.from_pretrained(MODEL_CKPT, language="Korean", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_CKPT, language="Korean", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_CKPT)

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_CKPT)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="korean", task="transcribe")

    logger.info("모델 로드 완료")

    df = load_data()

    processed_datasets = prepare_datasets(df, feature_extractor, tokenizer)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load('cer')

    logger.info("Trainer 설정 중")
    trainer = Seq2SeqTrainer(
        args=TRAINING_ARGS,
        model=model,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["valid"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer, metric),
        processing_class=processor.feature_extractor,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    logger.info(f"평가 결과: {eval_results}")
    
    logger.info(f"모델 저장 중: {TRAINING_ARGS.output_dir}")
    trainer.save_model(TRAINING_ARGS.output_dir)
    processor.save_pretrained(TRAINING_ARGS.output_dir)
    logger.info("모델 저장 완료")

if __name__ == "__main__":
    main()
