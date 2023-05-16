from transformers import  AutoProcessor, T5Tokenizer
from datasets import load_dataset
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images,return_tensors='pt')
    output_ids = tokenizer(captions,return_tensors='pt',padding='max_length').input_ids
    inputs.update({'output_ids':output_ids})
    return inputs


def create_train_test():
     
    ds = load_dataset("lambdalabs/pokemon-blip-captions") ## Using default dataset, can edit later.
    ds = ds["train"].train_test_split(test_size=0.1,seed=42)
    train_ds = ds["train"]
    test_ds = ds["test"]
    train_ds.set_transform(transforms)
    test_ds.set_transform(transforms)
    
    return train_ds,test_ds
