import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertPreTrainedModel
from transformers import (AutoConfig, AutoTokenizer)


class TextEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

    def forward(self, inputs):
        _, hidden_output = self.bert(**inputs)
        return hidden_output


def data_loader(lines, tokenizer):
    data = tokenizer.batch_encode_plus(lines, max_length=10, padding='max_length', truncation=True)
    all_input_ids = torch.tensor(data['input_ids'], dtype=torch.long)
    all_attention_mask = torch.tensor(data['attention_mask'], dtype=torch.long)
    all_token_type_ids = torch.tensor(data['token_type_ids'], dtype=torch.long)
    return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)


def encode(lines):
    model_name_or_path = '/home/xyf/models/chinese/bert/pytorch/bert-base-chinese'
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = TextEncoder.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=config)

    tensorData = data_loader(lines, tokenizer)
    dataLoader = DataLoader(tensorData, batch_size=32)
    vector = []
    for batch in dataLoader:
        batch = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
        sentence_encode = model(batch)
        vector += sentence_encode.cpu().tolist()
    return vector


if __name__ == '__main__':
    lines = ['今天天气很好',
             '今天下雨了']
    encode(lines)
