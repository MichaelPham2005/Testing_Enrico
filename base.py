import random
from collections import defaultdict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class InstructBase(nn.Module):
    """
    Base class for preprocessing and dataloader
    """

    def __init__(self, base_config):
        super().__init__()

        # classes: label name [eg. PERS, ORG, ...]
        self.max_width = base_config.max_width
        self.base_config = base_config

    # Chuyển đổi một danh sách các span (thực thể) thành một cấu trúc dữ liệu cho phép tra cứu nhanh.
    # {(0, 2): 1, (4, 4): 1}
    def get_dict(self, spans): 
        dict_tag = defaultdict(int)
        for span in spans:
            dict_tag[(span[0], span[1])] = 1
        return dict_tag

    def preprocess_spans(self, tokens, ner, rel):
        # Giới hạn độ dài của chuỗi token
        max_len = self.base_config.max_len

        if len(tokens) > max_len:
            length = max_len
            tokens = tokens[:max_len]
        else:
            length = len(tokens)

        # Tạo danh sách các chỉ mục của các span (cặp chỉ số bắt đầu và kết thúc)
        spans_idx = []
        for i in range(length):
            spans_idx.extend([(i, i + j) for j in range(self.max_width)])

        # Tạo một từ điển tra cứu xem mỗi span có phải là entity (1) hay không (0)
        dict_lab = self.get_dict(ner) if ner else defaultdict(int)

        # 0 for null labels
        span_label = torch.LongTensor([dict_lab[i] for i in spans_idx]) # spans_label: [0, 1, 0, ...] (1 nếu là thực thể, 0 nếu không phải)
        spans_idx = torch.LongTensor(spans_idx) # chuyển đổi sang tensor dạng shape [n, 2]

        # mask for valid spans (Nếu span vượt quá độ dài của chuỗi, đánh dấu là không hợp lệ => -1)
        valid_span_mask = spans_idx[:, 1] > length - 1

        # mask invalid positions (spans_label: [0, 1, 0, ...] => spans_label: [0, 1, -1, ...] với -1 là vị trí không hợp lệ)
        span_label = span_label.masked_fill(valid_span_mask, -1)

        return {
            'tokens': tokens,
            'span_idx': spans_idx,
            'span_label': span_label,
            'seq_length': length,
            'entities': ner,
            'relations': rel,
        }

    # Khi DataLoader lấy nhiều sample từ dataset (thường là các dict), nó cần gom thành một batch.
    def collate_fn(self, batch_list, relation_types=None):

        # batch = [self.preprocess_spans(['tokenized_text'], b['spans'], b['relations'])
        # for b in batch_list]

        if relation_types is None: # Training mode

            negs = self.get_negatives(batch_list, 100)

            rel_to_id = []
            id_to_rel = []

            for b in batch_list:
                random.shuffle(negs)

                # negs = negs[:sampled_neg]
                max_neg_type_ratio = int(self.base_config.max_neg_type_ratio)

                if max_neg_type_ratio == 0:
                    # no negatives
                    neg_type_ratio = 0
                else:
                    neg_type_ratio = random.randint(0, max_neg_type_ratio)

                if neg_type_ratio == 0:
                    # no negatives
                    negs_i = []
                else:
                    negs_i = negs[:len(b['relations']) * neg_type_ratio]

                # this is the list of all possible relation types (positive and negative)
                types = list(set([el[-1] for el in b['relations']] + negs_i))
                # "Âm" không phải là "quan hệ giả hoàn toàn", mà là "quan hệ không đúng với sample hiện tại".


                # shuffle (every epoch)
                random.shuffle(types) # xáo trộn

                if len(types) != 0:
                    # prob of higher number shoul
                    # random drop
                    if self.base_config.random_drop:
                        num_ents = random.randint(1, len(types))
                        types = types[:num_ents] # Chỉ giữ lại num_ents loại quan hệ đầu tiên

                # maximum number of relation types
                types = types[:int(self.base_config.max_types)] # nếu số loại quan hệ lớn hơn max_types, chỉ giữ lại max_types loại đầu tiên

                # supervised training
                if "label" in b:
                    types = sorted(b["label"])

                class_to_id = {k: v for v, k in enumerate(types, start=1)}
                id_to_class = {k: v for v, k in class_to_id.items()}
                rel_to_id.append(class_to_id)
                id_to_rel.append(id_to_class)

            batch = [
                self.preprocess_spans(b["tokenized_text"], b["spans"], b["relations"]) for b in batch_list
            ]

        else: # Evaluation mode
            rel_to_id = {k: v for v, k in enumerate(relation_types, start=1)}
            id_to_rel = {k: v for v, k in rel_to_id.items()}
            batch = [
                self.preprocess_spans(b["tokenized_text"], b["spans"], b["relations"]) for b in batch_list
            ]

        # Hàm giúp bạn pad tensor có độ dài khác nhau để tạo tensor batch chuẩn, đưa về cùng kích thước.
        # padding_value = 0 => nếu span_idx có độ dài khác nhau, sẽ được pad (đệm) bằng 0
        span_idx = pad_sequence(
            [b['span_idx'] for b in batch], batch_first=True, padding_value=0
        )

        span_label = pad_sequence(
            [el['span_label'] for el in batch], batch_first=True, padding_value=-1
        )

        return {
            'seq_length': torch.LongTensor([el['seq_length'] for el in batch]),
            'span_idx': span_idx,
            'tokens': [el['tokens'] for el in batch],
            'span_mask': span_label != -1,
            'span_label': span_label,
            'entities': [el['entities'] for el in batch],
            'relations': [el['relations'] for el in batch],
            'classes_to_id': rel_to_id,
            'id_to_classes': id_to_rel,
        }

    @staticmethod
    def get_negatives(batch_list, sampled_neg=50):
        # Giả sử mỗi phần tử trong batch_list có trường 'relations' chứa các quan hệ
        # và mỗi quan hệ là một tuple hoặc danh sách với phần tử cuối là loại quan hệ
        # Ví dụ: {'relations': [(0, 1, 'relation_type1'), (2, 3, 'relation_type2'), ...]}
        # Lấy tất cả các loại quan hệ từ batch_list 
        rel_types = []
        for b in batch_list:
            types = set([el[-1] for el in b['relations']])
            rel_types.extend(list(types))

        # Lấy các loại quan hệ duy nhất
        ent_types = list(set(rel_types))

        # sample negatives
        random.shuffle(ent_types) # trộn ngẫu nhiên các loại quan hệ
        return ent_types[:sampled_neg]

    def create_dataloader(self, data, relation_types=None, **kwargs):
        return DataLoader(data, collate_fn=lambda x: self.collate_fn(x, relation_types), **kwargs)
