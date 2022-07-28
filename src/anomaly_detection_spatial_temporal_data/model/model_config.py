from transformers.configuration_utils import PretrainedConfig

class TaddyConfig(PretrainedConfig):

    def __init__(
        self,
        config,
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
        self.save_directory = config['save_directory']
        
class ElandConfig():
    def __init__(
        self,
        config
    ):
        self.dim_feats = config['dim_feats']
        self.hidden_size = config['hidden_size']
        self.n_layers = config['n_layers']
        self.epochs = config['epochs']
        self.seed = config['seed']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.dropout = config['dropout']
        self.gnnlayer_type = config['gnnlayer_type']
        self.rnn_type = config['rnn_type']
        self.pretrain_bm = config['pretrain_bm']
        self.pretrain_nc = config['pretrain_nc']
        self.alpha =  config['alpha']
        self.bmloss_type = config['bmloss_type']
        self.batch_size = config['batch_size']
        self.base_pred = config['base_pred']
        self.name = config['name']
        self.device = config['device']
        self.log = config['log']
        self.save_directory = config['save_directory']    
