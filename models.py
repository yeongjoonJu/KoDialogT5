from __future__ import absolute_import, division, print_function

import torch, os, shutil, glob
import torch.nn as nn
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor

class T5forDialog(T5ForConditionalGeneration):
    def __init__(self, config):
        super(T5forDialog, self).__init__(config)
        
        self.config = config
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=0)

    def initialize_weights(self, modules):
        for module in modules:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def prepare_inputs_for_generation(self, input_ids,
                                      past=None, attention_mask=None,
                                      use_cache=None, encoder_outputs=None,
                                      **kwargs):
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"decoder_input_ids": input_ids,
                "past_key_values": past,
                "encoder_outputs": encoder_outputs,
                "attention_mask": attention_mask,
                "use_cache": use_cache}

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_attention_mask=None,
                decoder_input_ids=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                lm_labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_only=None,):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # span_loss, pred_spans, span_logits = 0, None, None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           inputs_embeds=inputs_embeds,
                                           return_dict=return_dict)

            if return_dict:
                encoder_hidden_states = encoder_outputs.last_hidden_state
            else:
                encoder_hidden_states = encoder_outputs[0]
        else:
            if isinstance(encoder_outputs, tuple):
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs.last_hidden_state

        if encoder_only:
            return encoder_outputs

        if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(lm_labels)

        if past_key_values is not None:
            assert lm_labels is None, "Decoder should not use cached key value states when training"
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                    inputs_embeds=decoder_inputs_embeds,
                                    past_key_values=past_key_values,
                                    attention_mask=decoder_attention_mask,
                                    encoder_hidden_states=encoder_hidden_states,
                                    encoder_attention_mask=attention_mask,
                                    use_cache=use_cache,
                                    return_dict=return_dict)

        sequence_output = decoder_outputs[0]

        sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        lm_loss = None
        if lm_labels is not None:
            lm_loss = self.lm_criterion(
                lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
        
        # for training
        if not return_dict:
            pred_lm = torch.argmax(lm_logits, dim=-1)
            outputs = (lm_loss, pred_lm, encoder_hidden_states)
        # for prediction
        else:
            outputs = Seq2SeqLMOutput(
                loss=lm_loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                encoder_attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)

        return outputs


class Runner(pl.LightningModule):
    def __init__(self, config, args, resized_vocab_size=None):
        super().__init__()

        model_class = T5forDialog
        self.model = model_class.from_pretrained(args.backbone, config=config)
        self.config = config
        self.args = args
        self.tokenizer = None
        
        if resized_vocab_size is not None:
            self.model.resize_token_embeddings(resized_vocab_size)
            self.config.vocab_size = resized_vocab_size
        
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if self.args.optimizer=="adafactor":
            optimizer = Adafactor(optimizer_grouped_parameters, lr=self.args.learning_rate, \
                                relative_step=False, scale_parameter=False, warmup_init=False)
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps)
        else:
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, self.args.warmup_steps, self.args.num_training_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch['enc_inputs'],
                            attention_mask=batch['enc_attn_mask'],
                            lm_labels=batch['resp_label'],
                            return_dict=False)
        
        loss = outputs[0]

        self.log("Training loss", 
                {"LM loss": loss.clone().detach(),})
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch['enc_inputs'],
                                attention_mask=batch['enc_attn_mask'],
                                lm_labels=batch['resp_label'],
                                return_dict=False)
        
        loss = outputs[0]

        # calculate losses
        return_dict = {"lm_loss":loss}
        if batch_idx % 20==0:
            context = batch["enc_inputs"][0].tolist()
            resp_pred = outputs[1][0].tolist()
            return_dict.update({"context":context, "resp_pred": resp_pred})

        return return_dict
        
    def validation_epoch_end(self, validation_step_outputs):
        ppl = 0.0
        contexts = []
        resp_preds = []
        for output in validation_step_outputs:
            ppl += output['lm_loss'].clone().detach()
            if "context" in output:
                contexts.append(output["context"])
                resp_preds.append(output["resp_pred"])

        ppl /= len(validation_step_outputs)
        ppl = torch.exp(ppl.clone().detach())

        self.log('val_PPL', ppl)
        self.text_log_step(contexts, resp_preds)
    
    def text_log_step(self, contexts, resps) -> None:
        columns = ['Context', "Resp_pred"]
        data = []
        for i, (c, r) in enumerate(zip(contexts, resps)):
            data.append([self.tokenizer.decode(c, clean_up_tokenization_spaces=True), \
                         self.tokenizer.decode(r, clean_up_tokenization_spaces=True) ])
            if i > 4:
                break
            
        self.logger.log_table(key='valid_text', columns=columns, data=data)


    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer