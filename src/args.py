from transformers import Seq2SeqTrainingArguments


TRAINING_ARGS = Seq2SeqTrainingArguments(
    output_dir="output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    num_train_epochs=5,
    gradient_checkpointing=True,
    fp16=False,
    eval_strategy="epoch",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    logging_steps=10,
    logging_strategy="steps",
    save_strategy="epoch",
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    push_to_hub=False,
    weight_decay=0.01,
)
