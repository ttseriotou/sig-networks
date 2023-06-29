from __future__ import annotations
import signatory
import torch
import torch.nn as nn
from nlpsig_networks.swnu import SWNU
from nlpsig_networks.ffn_baseline import FeedforwardNeuralNetModel


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
        embedding_dim: int, 
        hidden_dim_ffn: list[int] | int,
        output_dim: int, 
        dropout_rate: float, 
        augmentation_type: str = 'Conv1d', 
        hidden_dim_aug: list[int] | int | None = None,
        BiLSTM: bool = False,
        comb_method: str ='concatenation'):
        """
        SeqSigNet network for classification.
        
        Input data will have the size: [batch size, window size (w), all embedding dimensions (history + time + post), unit size (n)]
        Note: unit sizes will be in reverse chronological order, starting from the more recent and ending with the one further back in time.
        
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
        embedding_dim: int
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
        
        self.embedding_dim = embedding_dim #384
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
                "`comb_method` must be either 'concatenation' or 'gated_addition' "
                "or 'gated_concatenation' or 'scaled_concatenation'."
            )
        self.comb_method = comb_method

        # convolution
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1, 
            padding=1
        ).double()

        # alternative to convolution: using Augment from signatory 
        self.augment = signatory.Augment(
            in_channels=input_channels,
            layer_sizes = self.hidden_dim_aug + [output_channels],
            kernel_size=3,
            padding=1,
            stride=1,
            include_original=False,
            include_time=False
        ).double()

        # Non-linearity
        self.tanh1 = nn.Tanh()
 
        # signature window network unit to obtain feature set for FFN
        if isinstance(hidden_dim_swnu, int):
            hidden_dim_swnu = [hidden_dim_swnu]
        
        # Signatures and LSTMs for signature windows
        self.swnu = SWNU(input_size=output_channels,
                         hidden_dim=hidden_dim_swnu,
                         log_signature=log_signature,
                         sig_depth=sig_depth,
                         BiLSTM=BiLSTM).double()
    
        # signature without lift (for passing into BiLSTM)
        if log_signature:
            input_dim_lstmsig = signatory.logsignature_channels(in_channels=hidden_dim_swnu[-1], depth=sig_depth)
        else:
            input_dim_lstmsig = signatory.signature_channels(in_channels=hidden_dim_swnu[-1], depth=sig_depth)

        # BiLSTM
        self.lstm_sig2 = nn.LSTM(input_size=input_dim_lstmsig,
                                 hidden_size=hidden_dim_lstm,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True).double()

        # combination method
        if comb_method=='concatenation':
            # input dimensions for FFN
            input_dim = hidden_dim_lstm + self.embedding_dim + self.num_time_features
        elif comb_method=='gated_addition':
            input_gated_linear = hidden_dim_lstm + self.num_time_features
            if self.embedding_dim > 0:
                self.fc_scale = nn.Linear(input_gated_linear, self.embedding_dim)
                self.scaler = torch.nn.Parameter(torch.zeros(1, self.embedding_dim))
                #input dimensions for FFN
                input_dim = self.embedding_dim
            else:
                self.fc_scale = nn.Linear(input_gated_linear, input_gated_linear)
                self.scaler = torch.nn.Parameter(torch.zeros(1, input_gated_linear))
                #input dimensions for FFN
                input_dim = input_gated_linear
            # non-linearity
            self.tanh2 = nn.Tanh()
        elif comb_method=='gated_concatenation':
            # input dimensions for FFN
            input_dim = hidden_dim_lstm + self.embedding_dim + self.num_time_features
            # define the scaler parameter
            input_gated_linear = hidden_dim_lstm + self.num_time_features
            self.scaler1 = torch.nn.Parameter(torch.zeros(1,input_gated_linear))
        elif comb_method=='scaled_concatenation':
            input_dim = hidden_dim_lstm + self.embedding_dim + self.num_time_features
            # define the scaler parameter
            self.scaler2 = torch.nn.Parameter(torch.tensor([0.0]))
        
        # FFN for classification
        # make sure hidden_dim_ffn a list of integers
        if isinstance(hidden_dim_ffn, int):
            hidden_dim_ffn = [hidden_dim_ffn]
        self.hidden_dim_ffn = hidden_dim_ffn
        
        self.ffn = FeedforwardNeuralNetModel(input_dim=input_dim,
                                             hidden_dim=self.hidden_dim_ffn,
                                             output_dim=output_dim,
                                             dropout_rate=dropout_rate)

    def _unit_swnu(self,u):
        # convolution
        # u has dimensions [batch size, #posts (length of signal), embedding size]
        if self.augmentation_type == "Conv1d":
            out = torch.transpose(u, 1, 2)
            out = self.conv(out) 
            out = self.tanh1(out)
            out = torch.transpose(out, 1,2) #swap dimensions
        elif self.augmentation_type == "signatory":
            out = self.augment(out)
       
        # use SWNU to obtain feature set
        # out has dimensions [batch size, #posts (length of signal), embedding size]
        out = self.swnu(out)

        return out

    def forward(self, x):       
        # SWNU for each history window
        out = self._unit_swnu(x[:,:, :self.input_channels, 0])
        out = out.unsqueeze(1)
        for window in range(1,x.shape[3]):
            out_unit = self._unit_swnu(x[:,:,:self.input_channels,window])
            out_unit = out_unit.unsqueeze(1)
            out = torch.cat((out, out_unit), dim=1)
        
        # order sequences based on sequence length of input
        seq_lengths = torch.sum(torch.sum(torch.sum(x[:, :, :self.input_channels, :], 1) != 0, 1) != 0 , 1)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        out = out[perm_idx]
        out = torch.nn.utils.rnn.pack_padded_sequence(out, seq_lengths, batch_first=True)
        
        # BiLSTM that combines all deepsignet windows together
        _, (out, _) = self.lstm_sig2(out)
        out = out[-1, :, :] + out[-2, :, :]
        
        # reverse sequence padding
        inverse_perm = np.argsort(perm_idx)
        out = out[inverse_perm]
        
        # Combine Time Features and Last Post Embedding
        if self.comb_method=='concatenation':
            if self.num_time_features > 0:
                # concatenate any time features
                out = torch.cat((out, x[:, :, self.input_channels:(self.input_channels+self.num_time_features), 0].max(1)[0]), dim=1)
            if self.embedding_dim > 0:
                # concatenate current post embedding if provided
                out = torch.cat((out, x[:,0,(self.input_channels+self.num_time_features):, 0]), dim=1)
        elif self.comb_method=='gated_addition':
            if self.num_time_features > 0:
                # concatenate any time features
                out_gated = torch.cat((out, x[:, :, self.input_channels:(self.input_channels+self.num_time_features),0].max(1)[0]), dim=1)
            else:
                out_gated = out
            out_gated = self.fc_scale(out_gated.float())
            out_gated = self.tanh2(out_gated)
            out_gated = torch.mul(self.scaler, out_gated)
            if self.embedding_dim > 0:
                # add current post embedding if provided
                out = out_gated + x[:,0,(self.input_channels+self.num_time_features):, 0] 
            else:
                out = out_gated
        elif self.comb_method=='gated_concatenation':
            if self.num_time_features > 0:
                # concatenate any time features
                out_gated = torch.cat((out, x[:, :, self.input_channels:(self.input_channels+self.num_time_features), 0].max(1)[0]), dim=1)
            else:
                out_gated = out
            out_gated = torch.mul(self.scaler1, out_gated) 
            if self.embedding_dim > 0:
                # add current post embedding if provided 
                out = torch.cat((out_gated, x[:, 0, (self.input_channels+self.num_time_features):, 0]), dim=1 )
            else:
                out = out_gated 
        elif self.comb_method=='scaled_concatenation':
            if self.num_time_features > 0:
                # concatenate any time features
                out_gated = torch.cat((out, x[:, :, self.input_channels:(self.input_channels+self.num_time_features), 0].max(1)[0]), dim=1)
            else:
                out_gated = out
            out_gated = self.scaler2 * out_gated  
            if self.embedding_dim > 0:
                # add current post embedding if provided 
                out = torch.cat((out_gated , x[:,0, (self.input_channels+self.num_time_features):, 0]) , dim=1 ) 
            else:
                out = out_gated
        
        # FFN
        out = self.ffm(out)

        return out
