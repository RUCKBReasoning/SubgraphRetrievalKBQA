import torch
import torch.nn as nn

import transformers
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, **model_kwargs):
        super().__init__(config)
        self.model_kwargs = model_kwargs
        self.encoder = BertModel(config)
        self.sim = Similarity(temp=self.model_kwargs['temp'])
        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                output_attentions=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # seq_num: query, postive, k*negative
        bsz, seq_num, seq_length = input_ids.size()

        # flatten input for encoder
        input_ids = input_ids.view(bsz*seq_num, seq_length)
        attention_mask = attention_mask.view(bsz*seq_num, seq_length)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(bsz*seq_num, seq_length)
        outputs = self.encoder(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions, return_dict=True)
        pooler_output = outputs.pooler_output
        pooler_output = pooler_output.view(bsz, seq_num, -1)
        query, targets = pooler_output[:, 0:1], pooler_output[:, 1:]
        # calculate
        cos_sim = self.sim(query, targets)
        labels = torch.zeros(cos_sim.size(0)).long().to(self.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)
        if not return_dict:
            output = (cos_sim,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=cos_sim,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
