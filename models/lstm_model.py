import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import os
import logging

logger = logging.getLogger("TradingBot.LSTMModel")

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=17, hidden_size=128, num_layers=3, dropout=0.3):
        """
        LSTM tabanlı tahmin modeli
        
        Parametreler:
        - input_size: Giriş özelliklerinin sayısı (varsayılan: 17 - fiyat, teknik göstergeler, gap ve seans bilgileri)
        - hidden_size: LSTM gizli katman boyutu
        - num_layers: LSTM katman sayısı
        - dropout: Dropout oranı
        """
        super(LSTMPredictor, self).__init__()
        
        self.input_size = input_size  # Input size'ı sınıf değişkeni olarak kaydet
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout  # Dropout'u sınıf değişkeni olarak kaydet
        
        # LSTM katmanı
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Dikkat mekanizması
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Gap ve seans bilgileri için özel katman
        self.gap_session_layer = nn.Sequential(
            nn.Linear(5, 8),  # 5 özellik: gap, gap_size, session_asia, session_europe, session_us
            nn.ReLU()
        )
        
        # Çıkış katmanları
        self.fc1 = nn.Linear(hidden_size + 8, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        """
        İleri yayılım
        
        Parametreler:
        - x: Giriş verisi [batch_size, sequence_length, input_size]
        
        Dönüş:
        - Tahmin edilen değer [batch_size, 1]
        """
        batch_size = x.size(0)
        
        # LSTM katmanı
        lstm_out, _ = self.lstm(x)
        
        # Son zaman adımındaki çıktı
        last_time_step = lstm_out[:, -1, :]
        
        # Gap ve seans bilgilerini ayır
        # Varsayılan olarak son 5 özellik: gap, gap_size, session_asia, session_europe, session_us
        gap_session_features = x[:, -1, -5:]
        gap_session_encoded = self.gap_session_layer(gap_session_features)
        
        # LSTM çıktısı ve gap/seans bilgilerini birleştir
        combined = torch.cat((last_time_step, gap_session_encoded), dim=1)
        
        # Tam bağlantılı katmanlar
        x = self.fc1(combined)
        x = torch.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out
        
    def train_model(self, train_sequences, train_targets, val_sequences=None, val_targets=None,
                   sample_weights=None, epochs=100, batch_size=32, learning_rate=0.001, patience=10, verbose=False):
        """
        LSTM modelini eğitir
        
        Parametreler:
        - train_sequences: Eğitim verileri
        - train_targets: Eğitim hedefleri
        - val_sequences: Doğrulama verileri (opsiyonel)
        - val_targets: Doğrulama hedefleri (opsiyonel)
        - sample_weights: Örnek ağırlıkları (opsiyonel)
        - epochs: Eğitim döngüsü sayısı
        - batch_size: Batch boyutu
        - learning_rate: Öğrenme oranı
        - patience: Erken durdurma için sabır değeri
        - verbose: İlerleme bilgisini göster/gösterme
        
        Dönüş:
        - Eğitim geçmişi (kayıplar)
        """
        # Eğitim moduna geç
        self.train()
        
        # Ağırlıkları kontrol et
        if sample_weights is not None:
            logger.debug(f"Ağırlıklı eğitim kullanılıyor. Ağırlık aralığı: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
        
        # Kayıp fonksiyonu - ağırlıklı eğitim için reduction='none' kullanıyoruz
        criterion = nn.MSELoss(reduction='none')
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Öğrenme oranı scheduler'ı
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # Eğitim geçmişi
        train_losses = []
        val_losses = []
        
        # Erken durdurma için değişkenler
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Veri boyutunu logla
        logger.debug(f"Eğitim veri boyutu: {train_sequences.shape}, Hedef boyutu: {train_targets.shape}")
        if val_sequences is not None:
            logger.debug(f"Doğrulama veri boyutu: {val_sequences.shape}, Hedef boyutu: {val_targets.shape}")
        
        # Eğitim döngüsü
        for epoch in range(epochs):
            # Karıştırma indeksleri
            indices = torch.randperm(train_sequences.shape[0])
            
            # Batch'lere böl
            total_loss = 0
            num_batches = 0
            
            for i in range(0, train_sequences.shape[0], batch_size):
                # Batch indekslerini al
                batch_indices = indices[i:i+batch_size]
                
                # Batch verilerini al
                batch_sequences = train_sequences[batch_indices]
                batch_targets = train_targets[batch_indices]
                
                # Ağırlıkları al (varsa)
                batch_weights = None
                if sample_weights is not None:
                    batch_weights = sample_weights[batch_indices]
                
                # Forward pass
                outputs = self(batch_sequences)
                
                # Kayıp hesapla
                loss = criterion(outputs, batch_targets)
                
                # Ağırlıkları uygula (varsa)
                if batch_weights is not None:
                    loss = loss * batch_weights.unsqueeze(1)
                
                # Ortalama kayıp
                loss = loss.mean()
                
                # Backward pass ve optimize et
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Ortalama eğitim kaybı
            avg_train_loss = total_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Doğrulama kaybı (varsa)
            if val_sequences is not None and val_targets is not None:
                with torch.no_grad():
                    self.eval()  # Değerlendirme moduna geç
                    val_outputs = self(val_sequences)
                    val_loss = criterion(val_outputs, val_targets).mean().item()
                    val_losses.append(val_loss)
                    self.train()  # Eğitim moduna geri dön
                    
                    # Scheduler'ı güncelle
                    scheduler.step(val_loss)
                    
                    # Erken durdurma kontrolü
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Her epoch sonunda logla ve eğer verbose=True ise ekrana yazdır
                    if verbose or epoch % 10 == 0 or epoch == epochs - 1:
                        message = f"Epoch {epoch+1}/{epochs}, Eğitim Kaybı: {avg_train_loss:.6f}, Doğrulama Kaybı: {val_loss:.6f}"
                        logger.debug(message)
                        if verbose:
                            print(message)
                    
                    # Erken durdurma
                    if patience_counter >= patience:
                        message = f"Erken durdurma: Epoch {epoch+1}/{epochs}"
                        logger.info(message)
                        if verbose:
                            print(message)
                        break
            else:
                # Her epoch sonunda logla ve eğer verbose=True ise ekrana yazdır
                if verbose or epoch % 10 == 0 or epoch == epochs - 1:
                    message = f"Epoch {epoch+1}/{epochs}, Eğitim Kaybı: {avg_train_loss:.6f}"
                    logger.debug(message)
                    if verbose:
                        print(message)
        
        # Son eğitim durumunu logla
        if val_sequences is not None and val_targets is not None:
            message = f"Eğitim tamamlandı. Son Eğitim Kaybı: {train_losses[-1]:.6f}, Son Doğrulama Kaybı: {val_losses[-1]:.6f}"
            logger.info(message)
            if verbose:
                print(message)
        else:
            message = f"Eğitim tamamlandı. Son Eğitim Kaybı: {train_losses[-1]:.6f}"
            logger.info(message)
            if verbose:
                print(message)
        
        return {'train_losses': train_losses, 'val_losses': val_losses if val_sequences is not None else None}
    
    def evaluate(self, sequences, targets):
        """
        Modeli değerlendirir
        
        Parametreler:
        - sequences: Değerlendirilecek diziler
        - targets: Gerçek hedefler
        
        Dönüş:
        - Ortalama kayıp
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            outputs = self(sequences)
            loss = nn.MSELoss()(outputs, targets)
        return loss.item()
    
    def save_checkpoint(self, filename):
        """
        Model durumunu kaydeder
        
        Parametreler:
        - filename: Kaydedilecek dosya adı
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }
        
        # Dosya yolunun dizinini kontrol et ve gerekirse oluştur
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Modeli kaydet
        torch.save(checkpoint, filename)
        print(f"Model başarıyla kaydedildi: {filename}")
        return filename
    
    def load_checkpoint(self, filename):
        """
        Model durumunu yükler
        
        Parametreler:
        - filename: Yüklenecek dosya adı
        """
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.load_state_dict(checkpoint['model_state_dict'])
            return True
        return False

    @staticmethod
    def load_model(filename):
        """
        Kaydedilmiş bir LSTM modelini yükler
        
        Parametreler:
        - filename: Yüklenecek model dosyasının yolu
        
        Dönüş:
        - Yüklenmiş LSTMPredictor modeli
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {filename}")
        
        try:
            checkpoint = torch.load(filename)
            
            # Checkpoint içeriğini kontrol et
            if 'model_state_dict' not in checkpoint:
                raise ValueError(f"Geçersiz model dosyası: 'model_state_dict' bulunamadı")
            
            # Parametre değerlerini kontrol et ve varsayılan değerler kullan
            hidden_size = checkpoint.get('hidden_size', 128)
            num_layers = checkpoint.get('num_layers', 3)
            input_size = checkpoint.get('input_size', 17)
            dropout = checkpoint.get('dropout', 0.3)
            
            # Eksik parametreleri logla
            if 'hidden_size' not in checkpoint or 'num_layers' not in checkpoint:
                print(f"Uyarı: Model dosyasında bazı parametreler eksik. Varsayılan değerler kullanılıyor: hidden_size={hidden_size}, num_layers={num_layers}")
            
            if 'input_size' not in checkpoint:
                print(f"Uyarı: Model dosyasında input_size parametresi bulunamadı. Varsayılan değer kullanılıyor: input_size={input_size}")
            
            if 'dropout' not in checkpoint:
                print(f"Uyarı: Model dosyasında dropout parametresi bulunamadı. Varsayılan değer kullanılıyor: dropout={dropout}")
            
            # Yeni bir model oluştur
            model = LSTMPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
            
            # Model durumunu yükle
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"Model başarıyla yüklendi: {filename}")
            logger.info(f"Model başarıyla yüklendi: {filename}")
            return model
        
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {str(e)}")
            logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
            raise Exception(f"Model yüklenirken hata oluştu: {str(e)}") 