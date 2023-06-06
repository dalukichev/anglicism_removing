import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel,GPT2Tokenizer
from itertools import zip_longest
import warnings
warnings.filterwarnings("ignore")


class GPTgen():
    def __init__(self, model_path, weights_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = '<pad>'
        self.prefixes = torch.load(weights_path)
        self.prefixes = [x.to(self.device) for x in self.prefixes]
    
    
    def repetition_penalty_uniform(self, input_ids, logits, penalty=2.):
            ids_score = torch.gather(logits, 1,input_ids)
            ids_score = torch.where(ids_score<0, ids_score*penalty, ids_score/penalty)
            out_tensor = torch.clone(logits)
            out_tensor.scatter_(1,input_ids,ids_score)
            return out_tensor 
        
        
    def top_k_p_sampling(self, logits,top_k=0,top_p=0.0,filter_value=-float('inf')):
            if top_k>0:
                idx_to_null = logits<torch.topk(logits,top_k).values[...,-1,None]
                logits[idx_to_null] = filter_value
            if top_p>0.0:
                sorted_logs,sorted_idx = torch.sort(logits,descending=True)
                cumulative_proba = torch.cumsum(F.softmax(sorted_logs,dim=-1),dim=-1)
                sorted_idx_to_null = cumulative_proba>top_p
                sorted_idx_to_null[...,1:] = sorted_idx_to_null[...,:-1].clone()
                sorted_idx_to_null[...,0] = 0
                idx_to_null = torch.zeros_like(logits,dtype=sorted_idx_to_null.dtype).scatter_(dim=-1,index=sorted_idx,src=sorted_idx_to_null)
                logits[idx_to_null] = filter_value
            return logits
        
        
    def generate_text(self,
                      prefix_emb,
                      use_attention_mask,
                      sentence_len=100,
                      temp=1,
                      top_k=0,
                      top_p=0.0,
                      repetition_penalty=1.,
                      early_stop_val=50,
                      ):
        all_tokens=[]
        past=None
        inp_embedding=torch.clone(prefix_emb)
        attention_mask=torch.clone(use_attention_mask)
        with torch.no_grad():
            for pos in range(sentence_len):
                out=self.model(inputs_embeds=inp_embedding,
                               attention_mask=attention_mask.to(self.device),
                               past_key_values=past)
                past=out.past_key_values
                logits=out.logits[:,-1,:]
                logits=self.top_k_p_sampling(logits,top_k=top_k,top_p=top_p)
                if early_stop_val>0:
                    pos_to_multiply=torch.where(torch.topk(logits,k=1).indices==202,early_stop_val,1)
                    logits[...,202]*=pos_to_multiply.squeeze(1)
                    logits = logits/temp
    
    
                if len(all_tokens)>0:
                    res_tensor = torch.cat(all_tokens,dim=1)
                    logits = self.repetition_penalty_uniform(res_tensor.to(self.device),
                                                             logits,penalty=repetition_penalty)
                probas = F.softmax(logits,dim=-1)
                next_token = torch.multinomial(probas,1, replacement=True).detach().cpu()
                all_tokens.append(next_token)
                inp_embedding = self.model.get_input_embeddings()(next_token.to(self.device))
                attention_mask = torch.cat([attention_mask,
                                            torch.ones(attention_mask.shape[0],1).to(self.device)],dim=1)
    
                res_tensor = torch.cat(all_tokens,dim=1)
                min_val = (res_tensor==202).sum(-1)
                min_val = min_val.min().item()
                if min_val>0:
                    break
                    
            return torch.cat(all_tokens,dim=1)
        
        
    @torch.no_grad()
    def generate(self,
                 texts,
                 sentence_len=100, 
                 temp=0.7, 
                 top_p=0.7,
                 top_k=-1,
                 repetition_penalty=1.0, 
                 early_stop_val=50):
        assert len(self.prefixes) - len(texts) == 1, 'Problem with number of prefixes'
        texts = [[t] if isinstance(t, str) else t for t in texts]
        result = []
        input_embed = []
        attention_mask = []
        for pref, text in zip_longest(self.prefixes, texts):
            input_embed.append(pref.repeat_interleave(len(texts[0]),dim=0))
            attention_mask.append(torch.ones(len(texts[0]),pref.shape[1]).to(self.device))
            if text != None:
                text_tokens=self.tokenizer(text,padding=True,return_tensors='pt').to(self.device)
                text_embedding=self.model.get_input_embeddings()(text_tokens['input_ids'])
                attention_mask.append(text_tokens['attention_mask'])
                input_embed.append(text_embedding)
        mid_embed=torch.cat(input_embed,dim=1)
        mid_attention_mask = torch.cat(attention_mask,dim=1)
        generated_text=self.generate_text(
                                        mid_embed,
                                        mid_attention_mask,
                                        sentence_len=sentence_len,
                                        temp=temp,
                                        top_k=top_k,
                                        top_p=top_p,
                                        repetition_penalty=repetition_penalty,
                                        early_stop_val=early_stop_val
                                        )
        mask=(generated_text.detach().cpu()==202).int().cumsum(dim=-1)
        mask=torch.where(mask>0,0,1)
        generated_text=generated_text.detach().cpu()*mask
        result.append(generated_text)
        g_texts = [self.tokenizer.batch_decode(x, skip_special_tokens=True) for x in result]
        return g_texts
