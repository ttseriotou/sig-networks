from __future__ import annotations
import signatory
import torch
import torch.nn as nn

class SeqSigNet(nn.Module):
    """
    BiLSTM of Deep Signature Neural Network Units for classification.
    """
    def __init__(
        self, 
        input_channels: int, 
        output_channels: int,
        num_time_features: int, 
        log_signature: bool,
        sig_depth: int, 
        hidden_dim_swnu: list[int] | int,
        hidden_dim_lstm: int,
        input_bert_dim: int, 
        hidden_dim_ffn: list[int] | int,
        output_dim: int, 
        dropout_rate: float, 
        augmentation_type: str = 'Conv1d', 
        hidden_dim_aug: list[int] | int |None=(),
        BiLSTM: bool = False,
        comb_method: str ='concatenation'):
        """
        SeqSigNet network for classification.
        Input data will have the size: [batch size, all embedding dimensions (history + time + post), window size (w), unit size (n)]

        Parameters
        ----------
        input_channels : int
            Dimension of the (dimensonally reduced) history embeddings that will be passed in. 
        output_channels : int
            Requested dimension of the embeddings after convolution layer.
        num_time_features : int
            Number of time features to add to FFN input. If none, set to zero.
        log_signature : bool
            Whether or not to use the log signature or standard signature.
        sig_depth : int
            The depth to truncate the path signature at.
        hidden_dim_swnu : list[int] | int
            Dimensions of the hidden layers in the SNWU blocks.
        hidden_dim_lstm : int
            Dimensions of the hidden layers in the final BiLSTM of SWNU units.
        input_bert_dim: int
            Dimensions of current BERT post embedding. Usually 384 or 768.
        hidden_dim_ffn: int
            Dimension of the hidden layers in the FFN.
        output_dim : int
            Dimension of the output layer in the FFN.
        dropout_rate : float
            Dropout rate in the FFN.      
        augmentation_type : str, optional
            Method of augmenting the path, by default "Conv1d".
            Options are:
            - "Conv1d": passes path through 1D convolution layer.
            - else: passes path through `Augment` layer from `signatory` package.
        hidden_dim_aug: list[int] | int | None = ()
            Dimensions of the hidden layers in the augmentation layer.
            Passed into `Augment` class from `signatory` package if
            `augmentation_type!='Conv1d'`, by default ().
        BiLSTM : bool, optional
            Whether or not a birectional LSTM is used in the SWNU units,
            by default False (unidirectional LSTM is used in this case).
        comb_method : str ='concatenation'
            Determines how to combine the path signature and embeddings,
            by default "gated_addition".
            Options are:
            - concatenation: concatenation of path signature and embedding vector
            - gated_addition: element-wise addition of path signature and embedding vector
            - gated_concatenation: concatenation of linearly gated path signature and embedding vector
            - scaled_concatenation: concatenation of single value scaled path signature and embedding vector
        """

        super(SeqSigNet, self).__init__()
        
        self.input_bert_dim = input_bert_dim #384
        self.num_time_features = num_time_features
        self.input_channels = input_channels

        if isinstance(hidden_dim_aug, int):
            hidden_dim_aug = [hidden_dim_aug]
        elif hidden_dim_aug is None:
            hidden_dim_aug = []
        elif (hidden_dim_aug == ()):
            hidden_dim_aug = []
        self.hidden_dim_aug = hidden_dim_aug

        if augmentation_type not in ["Conv1d", "signatory"]:
            raise ValueError("`augmentation_type` must be 'Conv1d' or 'signatory'.")
        self.augmentation_type = augmentation_type

        if comb_method not in ["concatenation", "gated_addition", "gated_concatenation", "scaled_concatenation"]:
            raise ValueError(
                "`comb_method` must be either 'concatenation' or 'gated_addition' or 'gated_concatenation' or 'scaled_concatenation'."
            )
        self.comb_method = comb_method

        #Convolution
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size= 3,
            stride=1, 
            padding=1).double()

        # alternative to convolution: using Augment from signatory 
        self.augment = signatory.Augment(in_channels=input_channels,
                    layer_sizes = self.hidden_dim_aug + [output_channels],
                    kernel_size=3,
                    padding = 1,
                    stride = 1,
                    include_original=False,
                    include_time=False).double()

        #Non-linearity
        self.tanh1 = nn.Tanh()

        # signature window network unit to obtain feature set for FFN
        if isinstance(hidden_dim_swnu, int):
            hidden_dim_swnu = [hidden_dim_swnu]
        
        #Signatures and LSTMs for signature windows
        self.swnu = SWNU(input_size=output_channels,
                         hidden_dim=hidden_dim_swnu,
                         log_signature=log_signature,
                         sig_depth=sig_depth,
                         BiLSTM=BiLSTM).double()

        # signature without lift (for passing into BiLSTM)
        mult = 2 if BiLSTM else 1
        if log_signature:
            input_dim_lstmsig = signatory.logsignature_channels(in_channels=mult*hidden_dim_swnu[-1], depth=sig_depth)
        else:
            input_dim_lstmsig = signatory.signature_channels(in_channels=mult*hidden_dim_swnu[-1], depth=sig_depth)

        #BiLSTM
        self.lstm_sig2 = nn.LSTM(input_size=input_dim_lstmsig, hidden_size=hidden_dim_lstm, num_layers=1, batch_first=True, bidirectional=True).double()

        #combination method
        if comb_method=='concatenation':
            input_dim = hidden_dim_lstm + self.input_bert_dim + self.num_time_features
        elif comb_method=='gated_addition':
            input_dim = self.input_bert_dim
            input_gated_linear = signatory.logsignature_channels(hidden_dim_lstm, sig_depth) + self.num_time_features
            self.fc_scale = nn.Linear(input_gated_linear, self.input_bert_dim)
            #define the scaler parameter
            self.scaler = torch.nn.Parameter(torch.zeros(1,self.input_bert_dim))
        elif comb_method=='gated_concatenation':
            input_gated_linear = signatory.logsignature_channels(hidden_dim_lstm, sig_depth) + self.num_time_features
            input_dim = self.input_bert_dim + input_gated_linear
            #define the scaler parameter
            self.scaler1 = torch.nn.Parameter(torch.zeros(1,input_gated_linear))
        elif comb_method=='scaled_concatenation':
            input_dim = signatory.logsignature_channels(hidden_dim_lstm, sig_depth) + self.input_bert_dim + self.num_time_features 
            #define the scaler parameter
            self.scaler2 = torch.nn.Parameter(torch.tensor([0.0]))
        
        if isinstance(hidden_dim_ffn, int):
            hidden_dim_ffn = [hidden_dim_ffn]
        self.hidden_dim_ffn = hidden_dim_ffn
        # FFN: input layer
        self.ffn_input_layer = nn.Linear(input_dim, self.hidden_dim_ffn[0])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        input_hidden_dim = self.hidden_dim_ffn[0]
        
        # FFN: hidden layers
        self.ffn_linear_layers = []
        self.ffn_non_linear_layers = []
        self.dropout_layers = []
        for l in range(1,len(self.hidden_dim_ffn)):
            self.ffn_linear_layers.append(nn.Linear(input_hidden_dim, self.hidden_dim_ffn[l]))
            self.ffn_non_linear_layers.append(nn.ReLU())
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            input_hidden_dim = self.hidden_dim_ffn[l]
        
        self.ffn_linear_layers = nn.ModuleList(self.ffn_linear_layers)
        self.ffn_non_linear_layers = nn.ModuleList(self.ffn_non_linear_layers)
        self.dropout_layers = nn.ModuleList(self.dropout_layers)
        
        # FFN: readout
        self.ffn_final_layer = nn.Linear(input_hidden_dim, output_dim)

    def _unit_swnu(self,u):
        
        # convolution
        if self.augmentation_type == "Conv1d":
            out = self.conv(u) #get only the path information
            out = self.tanh1(out)
            out = torch.transpose(out, 1,2) #swap dimensions
        elif self.augmentation_type == "signatory":
            out = self.augment(torch.transpose(u,1,2))
       
        # use SWNU to obtain feature set
        out = self.swnu(out)
        return out

    def forward(self, x):
       
        #SWNU for each history window
        out = self._unit_swnu(x[:,:self.input_channels, :, -1])
        out = out.unsqueeze(1)
        for window in range(x.shape[3]-1,0, -1):
            out_unit = self._unit_swnu(x[:,:self.input_channels, :, window-1])
            out_unit = out_unit.unsqueeze(1)
            out = torch.cat((out, out_unit), dim=1)
 

        #BiLSTM that combines all deepsignet windows together
        _, (out, _) = self.lstm_sig2(out)
        out = out[-1, :, :] + out[-2, :, :]

        #Combine Time Features and Last Post Embedding
        if self.comb_method=='concatenation':
            out = torch.cat((out, x[:, self.input_channels:(self.input_channels+self.num_time_features),:, 0].max(2)[0], x[:,(self.input_channels+self.num_time_features):, 0, 0]), dim=1)
        elif self.comb_method=='gated_addition':
            out_gated = torch.cat((out, x[:, self.input_channels:(self.input_channels+self.num_time_features),:, 0].max(2)[0]), dim=1)
            out_gated = self.fc_scale(out_gated.float())
            out_gated = self.tanh1(out_gated)
            out_gated = torch.mul(self.scaler, out_gated)
            #concatenation with bert output
            out = out_gated + x[:,(self.input_channels+self.num_time_features):, 0]
        elif self.comb_method=='gated_concatenation':
            out_gated = torch.cat((out, x[:, self.input_channels:(self.input_channels+self.num_time_features),:, 0].max(2)[0]), dim=1)
            out_gated = torch.mul(self.scaler1, out_gated)  
            #concatenation with bert output
            out = torch.cat((out_gated, x[:,(self.input_channels+self.num_time_features):, 0, 0]), dim=1 ) 
        elif self.comb_method=='scaled_concatenation':
            out_gated = torch.cat((out, x[:, self.input_channels:(self.input_channels+self.num_time_features),:, 0].max(2)[0]), dim=1)
            out_gated = self.scaler2 * out_gated  
            #concatenation with bert output
            out = torch.cat((out_gated , x[:,(self.input_channels+self.num_time_features):, 0, 0]) , dim=1 ) 

        # FFN: input layer
        out = self.ffn_input_layer(out.float())
        out = self.relu(out)
        out = self.dropout(out)
        
        # FFN: hidden layers    
        for l in range(1,len(self.hidden_dim_ffn)):
            out = self.ffn_linear_layers[l-1](out)
            out = self.ffn_non_linear_layers[l-1](out)
            out = self.dropout_layers[l-1](out)

        # FFN: readout
        out = self.ffn_final_layer(out)

        return out
        
########################