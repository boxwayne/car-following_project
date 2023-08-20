import torch
from torch import nn
from math import pi
from modules import Encoder, Decoder


class Model(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_freq):
        super().__init__()
        self.num_freq = num_freq
        self.dim_freq_embedding = 1 + 2 * num_freq
        self.encoder = Encoder(dim_in, dim_hidden)
        self.decoder = Decoder(dim_hidden + self.dim_freq_embedding)

    def frequency_encoding(self, value, num_freq):  # value.size = (batch_size, 1)
        encoding_vector = []
        encoding_vector.append(value)
        for i in range(num_freq):
            encoding_vector.append(torch.sin(2. ** i * pi * value))
            encoding_vector.append(torch.cos(2. ** i * pi * value))
        encoding_output = torch.cat(encoding_vector, 1)
        assert encoding_output.shape[0] == value.shape[0]
        assert encoding_output.shape[1] == self.dim_freq_embedding
        return encoding_output

    def forward(self, states_histo, value_style, lead_speed):
        encoded_embedding = self.encoder(states_histo)
        style_embedding = self.frequency_encoding(value_style, self.num_freq)
        total_embedding = torch.cat((encoded_embedding, style_embedding), 1)
        predict_self_speed = self.decoder(lead_speed, total_embedding)
        return predict_self_speed
