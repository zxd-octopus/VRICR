
class TransformerOption:
    dimension_key = 64
    dimension_val = 64
    dimension_que = 64
    num_head = 8
    topic_num_layers = 3
    dropout = 0.1
    n_warmup_steps = 1000
    dimension_hidden = 2048
    graph_num_layers = 1
    num_layers = 12
    decode_max_len = 30
    beam_width = 5

    @staticmethod
    def update(attr, value):
        setattr(TransformerOption, attr, value)