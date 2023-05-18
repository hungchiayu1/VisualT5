
from typing import Tuple
import torch
from torch import nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
        
class VT5(nn.Module):
    
    
    def __init__(self,t5,tokenizer,vision_model,image_emb_size=512,prefix_length=10):
        super().__init__()
        self.t5 = t5
        self.tokenizer = tokenizer
        self.t5_embedding_size = t5.get_input_embeddings().embedding_dim
        self.image_emb_size = image_emb_size
        self.prefix_length = prefix_length
        self.vision_model = vision_model
        ## This is the mapping networks that projects the image embedding space to the language model vector space
        self.prefix_projection = MLP((self.image_emb_size, (self.t5_embedding_size * prefix_length) // 2,
                                     self.t5_embedding_size * prefix_length))
        
    def forward(self,pixel_values,output_ids):
   
        image_embeds = self.vision_model(pixel_values).image_embeds
        
        mapped_embedding = self.prefix_projection(image_embeds).view(-1,self.prefix_length,self.t5_embedding_size)
      
        ##concat_embedding = torch.cat([text_embedding,mapped_embedding],axis=1)
        
        output_ids[output_ids == self.tokenizer.pad_token_id] = -100 ## Do not compute loss w.r.t pad tokens
       
        outputs = self.t5(inputs_embeds=mapped_embedding,labels=output_ids)
        
        return outputs
    
    def generate_caption(self,pixel_values):
        
        image_embeds = self.vision_model(pixel_values).image_embeds
        mapped_embedding = self.prefix_projection(image_embeds).view(-1,self.prefix_length,self.t5_embedding_size)
        
        output_tokens = self.t5.generate(inputs_embeds=mapped_embedding)
        caption = self.tokenizer.decode(output_tokens[0],skip_special_tokens=True)
        
        return caption
        
        
        
  
