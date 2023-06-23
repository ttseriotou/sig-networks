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
        sig_depth: int, 
        hidden_dim_swnu: list[int] | int,
        hidden_dim_lstm: int,
        input_bert_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        dropout_rate: float, 
        augmentation_tp: str = 'Conv1d', 
        augmentation_layers: list[int] | int |None=(),
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
        sig_depth : int
            The depth to truncate the path signature at.
        hidden_dim_swnu : list[int] | int
            Dimensions of the hidden layers in the SNWU blocks.
        hidden_dim_lstm : int
            Dimensions of the hidden layers in the final BiLSTM of SWNU units.
        input_bert_dim: int
            Dimensions of current BERT post embedding. Usually 384 or 768.
        hidden_dim: int
            Dimension of the hidden layers in the FFN.
        output_dim : int
            Dimension of the output layer in the FFN.
        dropout_rate : float
            Dropout rate in the FFN.      
        augmentation_tp : str, optional
            Method of augmenting the path, by default "Conv1d".
            Options are:
            - "Conv1d": passes path through 1D convolution layer.
            - else: passes path through `Augment` layer from `signatory` package.
        augmentation_layers: list[int] | int | None = ()
            Dimensions of the hidden layers in the augmentation layer.
            Passed into `Augment` class from `signatory` package if
            `augmentation_type!='Conv1d'`, by default ().
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
        self.input_channels = input_channels
        self.augmentation_tp = augmentation_tp 
        self.comb_method = comb_method
        self.input_bert_dim = input_bert_dim #384

        #Convolution
        self.conv = nn.Conv1d(input_channels, output_channels, 3, stride=1, padding=1).double()
        self.augment = signatory.Augment(in_channels=input_channels,
                    layer_sizes = augmentation_layers,
                    kernel_size=3,
                    padding = 1,
                    stride = 1,
                    include_original=False,
                    include_time=False).double()
        #Non-linearity
        self.tanh1 = nn.Tanh()
        
        #Signatures and LSTMs for signature windows
        self.swnu = SWNU(input_size=output_channels,
                         hidden_dim=hidden_dim_swnu,
                         log_signature=True,
                         sig_depth=sig_depth,
                         BiLSTM=False).double()

        """
        #Signature with lift
        self.signature1 = signatory.LogSignature(depth=sig_depth, stream=True)
        self.lstm_sig1 = nn.LSTM(input_size=input_dim_lstm, hidden_size=hidden_dim_swnu[-1], num_layers=1, batch_first=True, bidirectional=False).double()
        self.signature2 = signatory.LogSignature(depth=sig_depth, stream=False)
        """
        input_dim_lstmsig = signatory.logsignature_channels(hidden_dim_swnu[-1] ,sig_depth)

        self.lstm_sig2 = nn.LSTM(input_size=input_dim_lstmsig, hidden_size=hidden_dim_lstm, num_layers=1, batch_first=True, bidirectional=True).double()

        #combination method
        if comb_method=='concatenation':
            input_dim = hidden_dim_lstm + self.input_bert_dim + 1 
        elif comb_method=='gated_addition':
            input_dim = self.input_bert_dim
            input_gated_linear = signatory.logsignature_channels(hidden_dim_lstm, sig_depth) + 1
            self.fc_scale = nn.Linear(input_gated_linear, self.input_bert_dim)
            #define the scaler parameter
            self.scaler = torch.nn.Parameter(torch.zeros(1,self.input_bert_dim))
        elif comb_method=='gated_concatenation':
            input_gated_linear = signatory.logsignature_channels(hidden_dim_lstm, sig_depth) + 1
            input_dim = self.input_bert_dim + input_gated_linear
            #define the scaler parameter
            self.scaler1 = torch.nn.Parameter(torch.zeros(1,input_gated_linear))
        elif comb_method=='scaled_concatenation':
            input_dim = signatory.logsignature_channels(hidden_dim_lstm, sig_depth) + self.input_bert_dim + 1 
            #define the scaler parameter
            self.scaler2 = torch.nn.Parameter(torch.tensor([0.0]))

        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.relu1 = nn.ReLU()
        #Dropout
        self.dropout = nn.Dropout(dropout_rate)
        # Linear function 2: 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()
        # Linear function 3 (readout): 
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def _unit_deepsignet(self,u):  
        #SWNU execution to form each input unit in the final BiLSTM

        out = u
        #Convolution
        if (self.augmentation_tp == 'Conv1d'):
            out = self.conv(out) #get only the path information
            out = self.tanh1(out)
            out = torch.transpose(out, 1,2) #swap dimensions
        else:
            out = self.augment(torch.transpose(out,1,2))

        
        #Add time for signature
        if self.add_time:
            out = torch.cat((out, torch.transpose(u[:,self.input_channels:(self.input_channels+1), :], 1,2)), dim=2)
        
        #Signature
        out = self.signature1(out)
        out, (_, _) = self.lstm_sig1(out)
        #Signature
        out = self.signature2(out)
        return out

    def _unit_swnu(self,u):
        
        # convolution
        if self.augmentation_tp == "Conv1d":
            out = self.conv(u) #get only the path information
            out = self.tanh1(out)
            out = torch.transpose(out, 1,2) #swap dimensions
        elif self.augmentation_type == "signatory":
            out = self.augment(torch.transpose(u,1,2))
       
        # use SWNU to obtain feature set
        out = self.swnu(out)
        return out

    def forward(self, x):
        """
        #SWNU for each history window
        out = self._unit_deepsignet(x[:,:self.input_channels, :, -1])
        out = out.unsqueeze(1)
        for window in range(x.shape[3]-1,0, -1):
            out_unit = self._unit_deepsignet(x[:,:self.input_channels, :, window-1])
            out_unit = out_unit.unsqueeze(1)
            out = torch.cat((out, out_unit), dim=1)

        """
        ################NEW CODE
        #SWNU for each history window
        out = self._unit_swnu(x[:,:self.input_channels, :, -1])
        out = out.unsqueeze(1)
        for window in range(x.shape[3]-1,0, -1):
            out_unit = self._unit_swnu(x[:,:self.input_channels, :, window-1])
            out_unit = out_unit.unsqueeze(1)
            out = torch.cat((out, out_unit), dim=1)
 
        #########################################

        #BiLSTM that combines all deepsignet windows together
        _, (out, _) = self.lstm_sig2(out)
        out = out[-1, :, :] + out[-2, :, :]

        #Combine Last Post Embedding
        if self.comb_method=='concatenation':
            out = torch.cat((out, x[:, self.input_channels:(self.input_channels+1),:, 0].max(2)[0], x[:,(self.input_channels+1):, 0, 0]), dim=1)
        elif self.comb_method=='gated_addition':
            out_gated = torch.cat((out, x[:, self.input_channels:(self.input_channels+1),:, 0].max(2)[0]), dim=1)
            out_gated = self.fc_scale(out_gated.float())
            out_gated = self.tanh1(out_gated)
            out_gated = torch.mul(self.scaler, out_gated)
            #concatenation with bert output
            out = out_gated + x[:,(self.input_channels+1):, 0]
        elif self.comb_method=='gated_concatenation':
            out_gated = torch.cat((out, x[:, self.input_channels:(self.input_channels+1),:, 0].max(2)[0]), dim=1)
            out_gated = torch.mul(self.scaler1, out_gated)  
            #concatenation with bert output
            out = torch.cat((out_gated, x[:,(self.input_channels+1):, 0, 0]), dim=1 ) 
        elif self.comb_method=='scaled_concatenation':
            out_gated = torch.cat((out, x[:, self.input_channels:(self.input_channels+1),:, 0].max(2)[0]), dim=1)
            out_gated = self.scaler2 * out_gated  
            #concatenation with bert output
            out = torch.cat((out_gated , x[:,(self.input_channels+1):, 0, 0]) , dim=1 ) 

        #FFN: Linear function 1
        out = self.fc1(out.float())
        # Non-linearity 1
        out = self.relu1(out)
        #Dropout
        out = self.dropout(out)

        #FFN: Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)
        #Dropout
        out = self.dropout(out)

        #FFN: Linear function 3 (readout)
        out = self.fc3(out)
        return out
        
########################