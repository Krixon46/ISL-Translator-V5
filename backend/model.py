import torch
import torch.nn as nn


class SignBiLSTM_V5(nn.Module):
    """
    SignBiLSTM_V5 Architecture matching the Kaggle trained V5 model.
    - BiLSTM: 2 layers, hidden_size=128 (output=256), dropout=0.3
    - Temporal Attention: Linear(256 -> 64) -> Tanh -> Linear(64 -> 1) -> Softmax across time
    - Context: Weighted sum of 32 frame outputs
    - Classifier: Linear(256 -> 64) -> ReLU -> Dropout(0.3) -> Linear(64 -> 9)
    """

    def __init__(
        self,
        input_size=126,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        fc_hidden_size=64,
        num_classes=9
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, fc_hidden_size),
            nn.Tanh(),
            nn.Linear(fc_hidden_size, 1)
        )

        self.fc1 = nn.Linear(hidden_size * 2, fc_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, x):
        # Expected input shape: [batch, 32, 126]
        lstm_out, _ = self.lstm(x)

        # Attention weights along temporal dimension: [batch, 32, 1]
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # Context vector: [batch, 256]
        context = torch.sum(lstm_out * attn_weights, dim=1)

        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# Alias for backward compatibility
SignBiLSTM = SignBiLSTM_V5