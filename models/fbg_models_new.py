"""
FBG Spektrumu Analiz Modelleri (PyTorch)
1D-CNN, LSTM, Autoencoder ve Hibrit Mimariler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN1DModel(nn.Module):
    """1D-CNN modeli - Spektrum sınıflandırması"""
    
    def __init__(self, input_size=512, num_classes=4, dropout_rate=0.3):
        super(CNN1DModel, self).__init__()
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        
        # Conv blokları
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        self.drop3 = nn.Dropout(dropout_rate)
        
        # Fully connected katmanlar
        self.fc1 = nn.Linear(16, 128)
        self.drop_fc1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.drop_fc2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # Conv blok 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Conv blok 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Conv blok 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop_fc1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop_fc2(x)
        
        x = self.fc3(x)
        
        return x


class LSTMModel(nn.Module):
    """LSTM modeli - Zaman serisi analizi"""
    
    def __init__(self, input_size=512, num_classes=4, lstm_units=64, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.lstm_units = lstm_units
        
        # LSTM katmanları
        self.lstm1 = nn.LSTM(1, lstm_units, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.bn1 = nn.BatchNorm1d(lstm_units * 2)
        
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units // 2, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.bn2 = nn.BatchNorm1d(lstm_units)
        
        # Fully connected katmanlar
        self.fc1 = nn.Linear(lstm_units, 128)
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(2)  # (batch_size, input_size, 1)
        
        # LSTM 1
        x, _ = self.lstm1(x)  # (batch_size, input_size, lstm_units*2)
        x = x.transpose(1, 2)
        x = self.bn1(x)
        x = x.transpose(1, 2)
        
        # LSTM 2
        x, (h, c) = self.lstm2(x)
        x = h[-1]  # Son hidden state al
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        
        return x


class DenoisingAutoencoder(nn.Module):
    """Spektrum gürültü temizleme için Autoencoder"""
    
    def __init__(self, input_size=512, encoding_dim=64):
        super(DenoisingAutoencoder, self).__init__()
        self.input_size = input_size
        
        # Encoder
        self.enc_conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.enc_pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.enc_conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.enc_pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.enc_conv3 = nn.Conv1d(64, encoding_dim, kernel_size=3, padding=1)
        
        # Decoder
        self.dec_conv1 = nn.Conv1d(encoding_dim, 64, kernel_size=3, padding=1)
        self.dec_up1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.dec_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.dec_up2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.dec_conv3 = nn.Conv1d(128, 1, kernel_size=3, padding=1)
    
    def encode(self, x):
        x = self.enc_conv1(x)
        x = F.relu(x)
        x = self.enc_pool1(x)
        
        x = self.enc_conv2(x)
        x = F.relu(x)
        x = self.enc_pool2(x)
        
        x = self.enc_conv3(x)
        x = F.relu(x)
        
        return x
    
    def decode(self, x):
        x = self.dec_conv1(x)
        x = F.relu(x)
        x = self.dec_up1(x)
        
        x = self.dec_conv2(x)
        x = F.relu(x)
        x = self.dec_up2(x)
        
        x = self.dec_conv3(x)
        x = torch.sigmoid(x)
        
        return x
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        
        return decoded.squeeze(1)


class CNNLSTMHybrid(nn.Module):
    """CNN-LSTM Hibrit Modeli"""
    
    def __init__(self, input_size=512, num_classes=4):
        super(CNNLSTMHybrid, self).__init__()
        
        # CNN özellik çıkarıcısı
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(0.2)
        
        # LSTM
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Fully connected
        self.fc1 = nn.Linear(128, 128)
        self.drop3 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.drop4 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # CNN
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)
        
        # x shape: (batch_size, 32, time_steps)
        x = x.transpose(1, 2)  # (batch_size, time_steps, 32)
        
        # LSTM
        x, (h, c) = self.lstm(x)
        x = h[-1]  # Son hidden state
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop3(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop4(x)
        
        x = self.fc3(x)
        
        return x


class AttentionLayer(nn.Module):
    """Self-Attention katmanı"""
    
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Attention ağırlıkları
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
        attention_weights = F.softmax(scores, dim=-1)
        
        # Attention uygulanmış çıkış
        output = torch.matmul(attention_weights, V)
        
        return output


class AttentiveSpecExLSTM(nn.Module):
    """Dikkat Mekanizmalı Spektrum Extrapolator LSTM"""
    
    def __init__(self, input_size=512, num_classes=4, lstm_units=128):
        super(AttentiveSpecExLSTM, self).__init__()
        
        # LSTM
        self.lstm1 = nn.LSTM(1, lstm_units, batch_first=True, bidirectional=True)
        
        # Attention
        self.attention = AttentionLayer(lstm_units * 2)
        
        # 2. LSTM
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units // 2, batch_first=True)
        
        # Fully connected
        self.fc1 = nn.Linear(lstm_units // 2, 256)
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.drop3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(2)  # (batch_size, input_size, 1)
        
        # LSTM 1
        x, _ = self.lstm1(x)
        
        # Attention
        x = self.attention(x)
        
        # LSTM 2
        x, (h, c) = self.lstm2(x)
        x = h[-1]  # Son hidden state
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        
        return x


class EnsembleModel(nn.Module):
    """Ensemble Modeli - CNN + LSTM + Attention"""
    
    def __init__(self, input_size=512, num_classes=4):
        super(EnsembleModel, self).__init__()
        
        # CNN branch
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # LSTM branch
        self.lstm1 = nn.LSTM(1, 64, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm2 = nn.LSTM(128, 32, batch_first=True)
        
        # Fully connected (birleştirilmiş)
        self.fc1 = nn.Linear(64, 256)
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.drop3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x_cnn = x.unsqueeze(1)  # (batch_size, 1, input_size)
        x_lstm = x.unsqueeze(2)  # (batch_size, input_size, 1)
        
        # CNN branch
        cnn_out = self.conv1(x_cnn)
        cnn_out = self.bn1(cnn_out)
        cnn_out = F.relu(cnn_out)
        cnn_out = self.pool1(cnn_out)
        
        cnn_out = self.conv2(cnn_out)
        cnn_out = F.relu(cnn_out)
        cnn_out = self.avgpool(cnn_out)
        cnn_out = cnn_out.squeeze(2)
        
        # LSTM branch
        lstm_out, _ = self.lstm1(x_lstm)
        lstm_out, (h, c) = self.lstm2(lstm_out)
        lstm_out = h[-1]
        
        # Şubeler birleştir
        merged = torch.cat([cnn_out, lstm_out], dim=1)
        
        # Fully connected
        x = self.fc1(merged)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        
        return x


def create_model(model_name='cnn1d', input_size=512, num_classes=4, device='cpu'):
    """Model oluştur ve device'a taşı"""
    
    models = {
        'cnn1d': CNN1DModel(input_size, num_classes),
        'lstm': LSTMModel(input_size, num_classes),
        'autoencoder': DenoisingAutoencoder(input_size),
        'cnn_lstm': CNNLSTMHybrid(input_size, num_classes),
        'attention_lstm': AttentiveSpecExLSTM(input_size, num_classes),
        'ensemble': EnsembleModel(input_size, num_classes)
    }
    
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' bulunamadı. Seçenekler: {list(models.keys())}")
    
    model = models[model_name]
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    print("PyTorch Model Mimarileri Test Ediliyor...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    input_size = 512
    batch_size = 32
    
    # Test verisi
    x_test = torch.randn(batch_size, input_size).to(device)
    
    models_to_test = [
        'cnn1d', 'lstm', 'cnn_lstm', 'attention_lstm', 'ensemble'
    ]
    
    for model_name in models_to_test:
        model = create_model(model_name, input_size, 4, device)
        param_count = sum(p.numel() for p in model.parameters())
        
        with torch.no_grad():
            output = model(x_test)
        
        print(f"{model_name.upper():15} -> Parametreler: {param_count:>10,} | Çıkış: {output.shape}")
