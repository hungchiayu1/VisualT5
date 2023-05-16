
from evaluate import load
import argparse
import torch
from torch import nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    T5Tokenizer, 
    T5ForConditionalGeneration,
    CLIPVisionModelWithProjection,
    AutoProcessor
)
from VT5 import VT5
from prepare_dataset import create_train_test

clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

wer = load("wer")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int,default=8)
    parser.add_argument('--gradient_accumulation_steps',type=int,default=1)
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    args = parser.parse_args()
    
    vt5 = VT5(t5,tokenizer,clip,image_emb_size=512,prefix_length=args.prefix_length)
    train_ds,test_ds = create_train_test()
    print("---- FINISH LOADING DATASET -----")
    training_args = TrainingArguments(
        output_dir=f"{args.out_dir}",
        learning_rate=5e-5,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=50,
        remove_unused_columns=False,
        logging_steps=50
    )

    trainer = Trainer(
        model=vt5,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    print("----- TRAINING START -----")
    trainer.train()
    
if __name__ == '__main__':
    main()

    
