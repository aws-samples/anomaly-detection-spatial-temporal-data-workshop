from transformers.configuration_utils import PretrainedConfig

class TaddyConfig(PretrainedConfig):

    def __init__(
        self,
        config,
        k=5,
        max_hop_dis_index = 100,
        max_inti_pos_index = 100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        hidden_act="gelu",
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.3,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_decoder=False,
        batch_size = 256,
        window_size = 1,
        weight_decay = 5e-4,
        **kwargs
    ):
        super(TaddyConfig, self).__init__(**kwargs)
        self.max_hop_dis_index = config['max_hop_dis_index']
        self.max_inti_pos_index = config['max_inti_pos_index']
        self.k = config['neighbor_num']
        self.hidden_size = config['embedding_dim']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_attention_heads = config['num_attention_heads']
        self.hidden_act = config['hidden_act']
        self.intermediate_size = config['embedding_dim']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.initializer_range = config['initializer_range']
        self.layer_norm_eps = config['layer_norm_eps']
        self.is_decoder = config['is_decoder']
        self.batch_size = config['batch_size']
        self.window_size = config['window_size']
        self.weight_decay = config['weight_decay']
        self.lr = config['lr']
        self.max_epoch = config['max_epoch']
        self.spy_tag = config['spy_tag']
        self.print_feq = config['print_feq']
        self.seed = config['seed']
