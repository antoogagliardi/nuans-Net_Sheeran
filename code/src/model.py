import re
import random
from typing import Union
from pprint import pprint
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import pytorch_lightning as pl

from torchmetrics.classification import BinaryAccuracy

from transformers import pipeline
from transformers import BertModel, BertTokenizer, BertForNextSentencePrediction

# NLTK
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')



# BERT Song Writer
class BERT_SongWriter(pl.LightningModule):
    def __init__(self, device):
        super(BERT_SongWriter, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        print(f"- pad_token_id: {self.tokenizer.pad_token_id} {self.tokenizer.pad_token}")
        print(f"- unk_token_id: {self.tokenizer.unk_token_id} {self.tokenizer.unk_token}")
        print(f"- cls_token_id: {self.tokenizer.cls_token_id} {self.tokenizer.cls_token}")
        print(f"- sep_token_id: {self.tokenizer.sep_token_id} {self.tokenizer.sep_token}")
        print(f"- mask_token_id: {self.tokenizer.mask_token_id} {self.tokenizer.mask_token}")

        self.NSP_model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased").to(device=device)

        # -- Metrics ---------------------------------------------
        self.nsp_accuracy = BinaryAccuracy().to(device=device)
    
    def get_masked(self, sentence:Union[str,torch.Tensor]):
        if isinstance(sentence, str):
            mask_idx = []
            s = re.split(" ", sentence)
            for i, tok in enumerate(s):
                if tok == self.tokenizer.mask_token:
                    mask_idx.append(i)
            return torch.LongTensor(mask_idx)
        if isinstance(sentence, torch.Tensor):
            mask_idx = (sentence == self.tokenizer.mask_token_id)       # Create a boolean mask using as a condition the mask id token
            nonzero_indices = torch.nonzero(mask_idx)                   # Find the indices of non-zero elements

            indices_list = nonzero_indices.transpose(dim0=-1, dim1=0)[1];
            indices_list = indices_list.reshape(shape=(indices_list.shape[-1]//2, 2))

            return indices_list
    
    def reconstruct_txt(self, masked_txt:str, mask_ids:list, mask_values:list):
        s = re.split(" ", masked_txt)
        for idx, mask in zip(mask_ids, mask_values):
            s[idx] = mask
        return " ".join(s)
    
    def choose_mask_element(self, maskLM_prediction, masked_idx, modality:str="argmax"):
        if modality == "sample":
            dist = Categorical(logits=F.log_softmax(maskLM_prediction.logits, dim=-1))
            pred_idxs = dist.sample()
        elif modality == "topk":
            kth_vals, kth_idx = F.log_softmax(maskLM_prediction.logits, dim=-1).topk(3, dim=-1)
            dist = Categorical(logits=kth_vals)
        elif modality == "argmax":
            pred_idxs = maskLM_prediction.logits.argmax(dim=-1)
        else:
            raise NotImplementedError("You've selected a wrong modality")
        
        mask_elem = torch.gather(input=pred_idxs, dim=1, index=masked_idx)
        return mask_elem

    def predict_next_sentence(self, ids, type, mask, nsp_labels):
        output = self.NSP_model(input_ids=ids, token_type_ids=type, attention_mask=mask, labels=nsp_labels)
        torch.mps.empty_cache()
        return output
    
    def predict_mask_tokens(self, ids, type, mask, mask_labels):
        output = self.maskLM_model(input_ids=ids, token_type_ids=type, attention_mask=mask, labels=mask_labels)
        torch.mps.empty_cache()
        return output

    def configure_optimizers(self):
        learning_rate = 0.001
        return torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, tok, tok_type, att, nsp_ground, mask_ground):
        nsp_out = self.predict_next_sentence(ids=tok, type=tok_type, mask=att, nsp_labels=nsp_ground)
        torch.mps.empty_cache()

        return nsp_out
    
    def training_step(self, batch, batch_idx):
        tokens = batch["tokens"]
        tokens_type = batch["tokens_type"]
        attention_mask = batch["attention_mask"]
        mask_ground_truth = batch["mask_ground_truth"]
        nsp_labels = batch["nsp_labels"]

        NSP_out = self(tok=tokens, tok_type=tokens_type, att=attention_mask,
                       nsp_ground=nsp_labels, mask_ground=mask_ground_truth)

        # --- Next Sentence Prediction Part of the Training -----------------------------------
        nsp_predictions = torch.argmax(NSP_out.logits, dim=-1).unsqueeze(dim=-1)
        nsp_accuracy = self.nsp_accuracy(preds=nsp_predictions, target=nsp_labels)
        # -------------------------------------------------------------------------------------
        
        self.log("train.loss", NSP_out.loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train.nsp_acc", nsp_accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": NSP_out.loss, "train.nsp_acc": nsp_accuracy}
    
    def validation_step(self, batch, batch_idx):
        tokens = batch["tokens"]
        tokens_type = batch["tokens_type"]
        attention_mask = batch["attention_mask"]
        mask_ground_truth = batch["mask_ground_truth"]      # ; print("mask ground truth: ", mask_ground_truth.shape)
        nsp_labels = batch["nsp_labels"]                    # ; print("nsp gold labels: ", nsp_labels.shape)
        
        NSP_out = self(tok=tokens, tok_type=tokens_type, att=attention_mask,
                       nsp_ground=nsp_labels, mask_ground=mask_ground_truth)
        # print("---- nsp output")
        # print("NSP prediction shape: ", NSP_out.logits.shape)
        # pprint(NSP_out)
 
        # --- Next Sentence Prediction Part of the Training -----------------------------------
        nsp_predictions = torch.argmax(NSP_out.logits, dim=-1).unsqueeze(dim=-1)
        # print("NSP FINAL PREDICTION SHAPE: ", nsp_predictions.shape)
        # print("NSP FINAL PREDICTION:\n ", nsp_predictions)
        nsp_accuracy = self.nsp_accuracy(preds=nsp_predictions, target=nsp_labels)
        # -------------------------------------------------------------------------------------
        
        self.log("valid.loss", NSP_out.loss, on_step=True, on_epoch=True, prog_bar=True)     
        self.log("valid.nsp_acc", nsp_accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return {"valid_loss": NSP_out.loss, "valid_nsp_acc": nsp_accuracy}
    
    def mask_predict(self, prompt_input:str, modality:str="argmax", device:str="cpu"):
        with torch.no_grad():
            masked_idx = self.get_masked(sentence=prompt_input).unsqueeze(dim=0).to(device=device); print("- [MASK] Elements Indices: ", masked_idx) 
            tokenized_prompt = self.tokenizer(prompt_input, padding="max_length", return_tensors="pt")           
            
            maskLM_pred = self.maskLM_model(**tokenized_prompt.to(device=device))                           # Get model predictions             
            print("-- Model Prediction --")
            print(" - Model Prediction Shape: ", maskLM_pred.logits.shape); pprint(maskLM_pred.logits)
            
            mask_pred = self.choose_mask_element(maskLM_pred, masked_idx=masked_idx, modality=modality)
            print("- Masked Elements: ", mask_pred)
            predicted_token = self.tokenizer.convert_ids_to_tokens(mask_pred[0])
            print("- [MASK] Elements Prediction: ", predicted_token)
            reconstructed_prompt = self.reconstruct_txt(prompt_input, mask_ids=masked_idx.squeeze().tolist(), mask_values=predicted_token)
            print("\n")
            print("ReconStructed Prompt: ", reconstructed_prompt)





# Lyrics Remixer
class LyricsRemixer():
    def __init__(self, model, device):
        super(LyricsRemixer, self).__init__()
        self.device = device
        
        self.stop_words = set(stopwords.words('english'))
        self.additional_stop_words = ['[INTRO]',
                                      '[VERSE]',
                                      '[PRE-CHORUS]', '[CHORUS]', '[POST-CHORUS]',
                                      '[REFRAIN]', '[BRIDGE]', '[INTERLUDE]', '[BREAKDOWN]', '[HOOK]',
                                      '[OUTRO]']
        self.remixer_model = model
        self.remixer_model.eval()                   # Put the model in evaluation mode
        
        TITLE_SUMMARIZATION_MODEL = "t5-large"
        self.title_summarizer = pipeline("summarization", model=TITLE_SUMMARIZATION_MODEL)

        self.bert_title_judger = BertModel.from_pretrained("bert-base-uncased")
        self.bert_title_judger.eval()


    def compose_new_song(self, test_in:dict, lines_dict:dict, truth_lines_dict:dict,
                         song_length:int=5, limit_candidate:int=20, device:str="cpu"):
        test_in_keys = list(test_in.keys())
        sentence = test_in_keys[random.randint(0, len(test_in_keys))]
        
        gold_candidates = truth_lines_dict[sentence]
        
        possible_next_candidates = lines_dict[sentence]
        len_candidates = len(possible_next_candidates)
        candidate_batch = [f"[CLS]{sentence}[SEP]{candidate}[SEP]" for candidate, i in zip(possible_next_candidates, range(len_candidates)) if i <= limit_candidate]

        final_gold_sentence = [sentence]
        final_sentence = [sentence]
        with tqdm(range(song_length), desc="Constructing New Song") as pbar:
            for _, i in zip(pbar, range(song_length)):
                torch.mps.empty_cache()
                tok_prompt = self.remixer_model.tokenizer(candidate_batch, padding="max_length", return_tensors="pt",
                                            add_special_tokens=False)
                tok_prompt.to(device=self.device)
                with torch.no_grad():
                    nsp_out = self.remixer_model.predict_next_sentence(ids=tok_prompt["input_ids"],
                                                                type=tok_prompt["token_type_ids"], mask=tok_prompt["attention_mask"],
                                                                nsp_labels=None)
                    torch.mps.empty_cache()
                _, idxs = torch.topk(nsp_out.logits[:,0], k=1)                          # get the top-k predictions and return the respective tokens
                next_sentence = re.sub("\n", "", possible_next_candidates[idxs])
                final_sentence.append(next_sentence)

                try:
                    gold_candidates = truth_lines_dict[next_sentence]
                    final_gold_sentence.append(gold_candidates)

                    possible_next_candidates = lines_dict[next_sentence]
                    len_candidates = len(possible_next_candidates)
                    candidate_batch = [f"[CLS] {next_sentence} [SEP] {candidate}" for candidate, i in zip(possible_next_candidates, range(len_candidates)) if i <= limit_candidate]
                    
                    pbar.update()
                except Exception as e:
                    print("Unexpected Error occurs during song lyric generation:", str(e))
                    print("Generation Interrupted...")
                    break

        final_sentence = "\n".join(final_sentence)
        final_gold_sentence = "\n".join(final_gold_sentence)

        return final_sentence, final_gold_sentence
    
    def create_song_title(self, song:str=None, min_l:int=1, max_l:int=8):
        if song == None: print("Error: No song has been provided")
        else:
            print("-- Filtering Song --")
            filtered_song = []
            split_new_song = re.split("\n", song)
            for l in split_new_song:
                tokens = re.split(" ", l); print(tokens)
                for w in tokens:
                    if w.strip("") in self.additional_stop_words: continue
                    if w.lower() not in self.stop_words: filtered_song.append(w)
            
            filtered_song = " ".join(filtered_song)
        
        print("\n")
        
        print(f"-- Summarizing Song ........")
        summary = self.title_summarizer(filtered_song, max_length=max_l, min_length=min_l, do_sample=False,
                                        no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3, repetition_penalty=3.5, num_beams=3)
        song_title = summary[0]["summary_text"]
        print(" - Title returned by the Summarizer: ", song_title)

        return song_title, song
    
    def judge_created_title(self, song:str, title:str):
        encod_new_song =  self.remixer_model.tokenizer(song, return_tensors="pt")
        encod_song_title =  self.remixer_model.tokenizer(title, return_tensors="pt")

        song_embed = self.bert_title_judger(**encod_new_song)
        title_emebed = self.bert_title_judger(**encod_song_title)
        
        # We're interested in the pooler_output of both the model output since it encoded
        #  the representation of [CLS] special token at the last hidden state of the encoder
        song_embed = song_embed.pooler_output
        title_emebed = title_emebed.pooler_output

        # Since BERT [CLS] special token encode the overall meaning of the entire sentence
        #  we can measure the cos similarity between generated title and song cls embeddings
        similarity = torch.cosine_similarity(x1=song_embed, x2=title_emebed, dim=-1)
        print("Song/Title Similarity: ", np.round(similarity.item(), decimals=4))
        