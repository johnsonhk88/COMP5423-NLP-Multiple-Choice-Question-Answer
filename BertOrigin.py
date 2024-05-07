
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

class BertOrigin(BertPreTrainedModel):
    """BERT model for multiple choice tasks. BERT + Linear

    Args:
        config: BertConfig 类对象， 以此创建模型
        num_choices: 选项数目，默认为 2.
    """

    def __init__(self, config, num_choices):
        super(BertOrigin, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Inputs:
            input_ids: [batch_size, num_choices, sequence_length]， 其中包含了词所对应的ids
            token_type_ids: 可选，[batch_size, num_choices, sequence_length]；0 表示属于句子 A， 1 表示属于句子 B
            attention_mask: 可选，[batch_size, num_choices, sequence_length]；区分 padding 与 token， 1表示是token，0 为padding
            labels: [batch_size], 其中数字在 [0, ..., num_choices]之间
        """
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        # flat_input_ids: [batch_size * num_choices, sequence_length]
        flat_token_type_ids = token_type_ids.view(
            -1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # flat_token_type_ids: [batch_size * num_choices, sequence_length]
        flat_attention_mask = attention_mask.view(
            -1, attention_mask.size(-1)) if attention_mask is not None else None
        # flat_attention_mask: [batch_size * num_choices, sequence_length]

        _, pooled_output = self.bert(
            flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        # pooled_output: [batch_size * num_choices, bert_dim]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # logits: [batch_size * num_choices, 1]
        reshaped_logits = logits.view(-1, self.num_choices)
        # reshaped_logits: [batch_size, num_choices]
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits
