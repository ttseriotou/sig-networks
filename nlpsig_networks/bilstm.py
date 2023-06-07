from __future__ import annotations
import torch
import torch.nn as nn

class BILSTM(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        hidden_dim_lstm: int,
        num_layers: int,
        bidirectional: bool,
        output_dim: int,
        dropout_rate: float
    ):
        """
        BiLSTM network

        Parameters
        ----------
        input_dim : int
            _description_
        hidden_dim_lstm : int
            _description_
        output_dim : int
            _description_
        dropout_rate : float
            _description_
        """
        super(BILSTM, self).__init__()
        self.hidden_dim_lstm1 = hidden_dim_lstm
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim_lstm,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional).double()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim_lstm, output_dim)
    
    def forward(self, x):
        # why is this part necessary?
        seq_lengths = torch.sum(x[:, :, 0] != 123, 1)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        x = x[perm_idx]

        #BiLSTM 1
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
        out, (_, _) = self.lstm1(x_pack)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        inverse_perm = np.argsort(perm_idx)
        out = out[inverse_perm]
        out = out[:, :, :self.hidden_dim_lstm1] + out[:, :, self.hidden_dim_lstm1:]

        out = self.dropout(out)

        #BiLSTM 2
        out = out[perm_idx]
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(out, seq_lengths, batch_first=True)
        outl, (out_h, _) = self.lstm2(x_pack)
        outl, _ = torch.nn.utils.rnn.pad_packed_sequence(outl, batch_first=True)
        outl = outl[inverse_perm]
        out = out_h[-1, :, :] + out_h[-2, :, :]
        out = out[inverse_perm]

        out = self.dropout(out)

        out = self.fc3(out.float())
        return out