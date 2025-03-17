import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import os
import logging
from tqdm.auto import tqdm
import torch.nn.functional as F
from config.config import MODEL_CONFIG  # MODEL_CONFIG'i import et

logger = logging.getLogger("TradingBot.LSTMModel")

class LSTMPredictor(nn.Module):
    def __init__(self, config):
        """
        LSTM tabanlı tahmin modeli
        
        Parametreler:
        - config: Model yapılandırma bilgileri
        """
        super(LSTMPredictor, self).__init__()
        
        # Config parametrelerini al
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # CUDA kullanılabilirliğini kontrol et
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Model {self.device} üzerinde çalışacak")
        
        # LSTM katmanı
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=True
        )
        
        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(
            self.hidden_size * 2,
            momentum=MODEL_CONFIG['batch_norm']['momentum'],
            eps=MODEL_CONFIG['batch_norm']['eps']
        )
        
        # Attention mekanizması
        attention_config = MODEL_CONFIG.get('attention', {
            'dims': [512, 128, 1],
            'dropout': 0.2
        })
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size * 2, attention_config['dims'][0]),
            nn.LayerNorm(attention_config['dims'][0]),
            nn.ReLU(),
            nn.Dropout(attention_config['dropout']),
            nn.Linear(attention_config['dims'][0], attention_config['dims'][1]),
            nn.Tanh(),
            nn.Linear(attention_config['dims'][1], attention_config['dims'][2])
        )
        
        # Fully connected katmanlar
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc_norm = nn.LayerNorm(self.hidden_size)
        self.fc_dropout = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.hidden_size, 1)
        
        # Modeli seçilen cihaza taşı
        self.to(self.device)
        
    def calculate_accuracy(self, outputs, targets, threshold=0.02):
        """Tahminlerin doğruluk oranını hesapla (±%2 tolerans ile)"""
        with torch.no_grad():
            # Yüzdesel değişim tahminlerinin doğruluğunu kontrol et
            correct = torch.abs(outputs - targets) <= threshold
            return correct.float().mean().item()
    
    def forward(self, x):
        """
        İleri yayılım
        
        Parametreler:
        - x: Giriş verisi [batch_size, sequence_length, input_size]
        
        Dönüş:
        - Tahmin edilen yüzdesel değişim [-0.1, 0.1] aralığında
        """
        # Veri doğrulama
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Giriş torch.Tensor olmalıdır, alınan: {type(x)}")
        if len(x.shape) != 3:
            raise ValueError(f"Giriş 3 boyutlu olmalıdır [batch, seq_len, features], alınan shape: {x.shape}")
        if x.shape[2] != self.input_size:
            raise ValueError(f"Giriş özellik sayısı {self.input_size} olmalıdır, alınan: {x.shape[2]}")
        
        # Veriyi GPU'ya taşı
        x = x.to(self.device)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Batch norm
        batch_norm_out = self.batch_norm(lstm_out.transpose(1, 2)).transpose(1, 2)
        
        # Attention
        attention_weights = self.attention(batch_norm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * batch_norm_out, dim=1)
        
        # Fully connected
        out = self.fc1(context_vector)
        out = self.fc_norm(out)
        out = F.relu(out)
        out = self.fc_dropout(out)
        out = self.fc2(out)
        
        # Çıktıyı [-0.1, 0.1] aralığına sınırla (±%10 değişim)
        return torch.tanh(out) * 0.1
    
    def train_model(self, train_sequences, train_targets, val_sequences=None, val_targets=None,
                   sample_weights=None, epochs=100, batch_size=32, learning_rate=0.001, patience=10, verbose=False,
                   interrupt_check=None, checkpoint_save=None, checkpoint_interval=5):
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
        - interrupt_check: Eğitimi durdurmak için kontrol fonksiyonu
        - checkpoint_save: Kontrol noktası kaydetme fonksiyonu
        - checkpoint_interval: Kontrol noktası kaydetme aralığı (epoch sayısı)
        
        Dönüş:
        - Eğitim geçmişi (kayıplar)
        """
        # Veri doğrulama
        if not isinstance(train_sequences, torch.Tensor):
            raise TypeError("train_sequences torch.Tensor olmalıdır")
        if not isinstance(train_targets, torch.Tensor):
            raise TypeError("train_targets torch.Tensor olmalıdır")
        if train_sequences.shape[0] != train_targets.shape[0]:
            raise ValueError("train_sequences ve train_targets boyutları eşleşmiyor")
        
        # Veri normalizasyonu
        def normalize_sequences(sequences):
            # Her bir özellik için ayrı normalizasyon
            normalized = torch.zeros_like(sequences)
            for i in range(sequences.shape[2]):  # Her bir özellik için
                feat = sequences[:, :, i]
                mean = feat.mean()
                std = feat.std()
                if std < 1e-8:
                    normalized[:, :, i] = feat - mean
                else:
                    normalized[:, :, i] = (feat - mean) / std
            return normalized

        # Eğitim ve doğrulama verilerini normalize et
        train_sequences = normalize_sequences(train_sequences)
        if val_sequences is not None:
            val_sequences = normalize_sequences(val_sequences)
        
        # Verileri GPU'ya taşı
        train_sequences = train_sequences.to(self.device)
        train_targets = train_targets.to(self.device)
        if val_sequences is not None:
            val_sequences = val_sequences.to(self.device)
            val_targets = val_targets.to(self.device)
        if sample_weights is not None:
            sample_weights = sample_weights.to(self.device)
        
        # Eğitim moduna geç
        self.train()
        
        # Ağırlıkları kontrol et
        if sample_weights is not None:
            logger.debug(f"Ağırlıklı eğitim kullanılıyor. Ağırlık aralığı: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
        
        # Kayıp fonksiyonu ve metrikler
        def custom_loss(outputs, targets, weights=None):
            """Özel kayıp fonksiyonu - MSE ve yön doğruluğu kombinasyonu"""
            # MSE loss
            mse_loss = F.mse_loss(outputs, targets, reduction='none')
            
            # Yön doğruluğu loss
            direction_correct = torch.sign(outputs) == torch.sign(targets)
            direction_loss = 1.0 - direction_correct.float()
            
            # Toplam kayıp
            total_loss = mse_loss + direction_loss * 0.1  # Yön kaybına %10 ağırlık ver
            
            if weights is not None:
                total_loss = total_loss * weights.unsqueeze(1)
            
            return total_loss.mean()
        
        # Optimizer ve scheduler
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=learning_rate,
            weight_decay=MODEL_CONFIG['training'].get('weight_decay', 0.001)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=MODEL_CONFIG['training'].get('reduce_lr_factor', 0.5),
            patience=MODEL_CONFIG['training'].get('reduce_lr_patience', 7),
            verbose=False
        )
        
        # Eğitim geçmişi
        train_losses = []
        val_losses = []
        
        # Erken durdurma için değişkenler
        best_val_loss = float('inf')
        patience_counter = 0
        
        try:
            # Eğitim döngüsü
            pbar = tqdm(range(epochs), desc="Eğitim", disable=not verbose)
            for epoch in pbar:
                # Check for interrupt first
                if interrupt_check and interrupt_check():
                    logger.info(f"Eğitim epoch {epoch+1}/{epochs}'de durduruldu")
                    if verbose:
                        print(f"\nEğitim epoch {epoch+1}/{epochs}'de kullanıcı tarafından durduruldu")
                    break
                
                # Bellek temizliği
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Karıştırma indeksleri
                indices = torch.randperm(train_sequences.shape[0])
                
                # Batch'lere böl
                total_loss = 0
                total_acc = 0
                num_batches = 0
                
                # Batch döngüsü
                for i in range(0, train_sequences.shape[0], batch_size):
                    # Check for interrupt in batch loop too
                    if interrupt_check and interrupt_check():
                        logger.info(f"Eğitim batch işlemi sırasında durduruldu (epoch {epoch+1})")
                        break
                    
                    # Batch verilerini hazırla
                    batch_indices = indices[i:i+batch_size]
                    batch_sequences = train_sequences[batch_indices]
                    batch_targets = train_targets[batch_indices]
                    batch_weights = sample_weights[batch_indices] if sample_weights is not None else None
                    
                    # Forward pass
                    outputs = self(batch_sequences)
                    
                    # Kayıp hesapla
                    loss = custom_loss(outputs, batch_targets, batch_weights)
                    
                    # Accuracy hesapla
                    batch_acc = self.calculate_accuracy(outputs, batch_targets)
                    total_acc += batch_acc
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient değerlerini kontrol et
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm):
                        raise ValueError("Gradient değerleri NaN")
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Belleği temizle
                    del batch_sequences, batch_targets, outputs, loss
                    if batch_weights is not None:
                        del batch_weights
                
                # Ortalama metrikler
                avg_train_loss = (total_loss / num_batches) * 0.1
                avg_train_acc = total_acc / num_batches
                train_losses.append(avg_train_loss)
                
                # Doğrulama
                if val_sequences is not None and val_targets is not None:
                    val_loss = self._validate(val_sequences, val_targets) * 0.1
                    val_acc = self._validate(val_sequences, val_targets, metric='accuracy')
                    val_losses.append(val_loss)
                    
                    # Scheduler'ı güncelle
                    scheduler.step(val_loss)
                    
                    # Progress bar'ı güncelle
                    pbar.set_postfix({
                        'train_loss': f'{avg_train_loss:.6f}',
                        'train_acc': f'{avg_train_acc:.2%}',
                        'val_loss': f'{val_loss:.6f}',
                        'val_acc': f'{val_acc:.2%}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    
                    # Save checkpoint if specified
                    if checkpoint_save and epoch > 0 and (epoch + 1) % checkpoint_interval == 0:
                        checkpoint_save(self.state_dict())
                    
                    # Erken durdurma kontrolü
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # Save best model if specified
                        if checkpoint_save:
                            checkpoint_save(self.state_dict())
                    else:
                        patience_counter += 1
                    
                    # Erken durdurma
                    if patience_counter >= patience:
                        message = f"Erken durdurma: Epoch {epoch+1}/{epochs}"
                        logger.info(message)
                        if verbose:
                            print(f"\n{message}")
                        break
                else:
                    # Sadece eğitim metriklerini göster
                    pbar.set_postfix({
                        'train_loss': f'{avg_train_loss:.6f}',
                        'train_acc': f'{avg_train_acc:.2%}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    
                    # Save checkpoint if specified
                    if checkpoint_save and epoch > 0 and (epoch + 1) % checkpoint_interval == 0:
                        checkpoint_save(self.state_dict())
            
            # Son durumu logla
            final_message = f"Eğitim tamamlandı. Son Eğitim Kaybı: {train_losses[-1]:.6f}"
            if val_losses:
                final_message += f", Son Doğrulama Kaybı: {val_losses[-1]:.6f}"
            logger.info(final_message)
            if verbose:
                print(f"\n{final_message}")
            
            return {'train_losses': train_losses, 'val_losses': val_losses if val_sequences is not None else None}
            
        except Exception as e:
            logger.error(f"Eğitim sırasında hata: {str(e)}")
            raise
        finally:
            # Belleği temizle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _validate(self, val_sequences, val_targets, batch_size=32, metric='loss'):
        """
        Doğrulama seti üzerinde değerlendirme yap
        """
        self.eval()
        total_metric = 0
        num_batches = 0
        criterion = nn.MSELoss(reduction='mean')
        
        with torch.no_grad():
            for i in range(0, len(val_sequences), batch_size):
                batch_sequences = val_sequences[i:i+batch_size]
                batch_targets = val_targets[i:i+batch_size]
                
                outputs = self(batch_sequences)
                
                if metric == 'loss':
                    batch_metric = criterion(outputs, batch_targets).item()
                elif metric == 'accuracy':
                    batch_metric = self.calculate_accuracy(outputs, batch_targets)
                
                total_metric += batch_metric
                num_batches += 1
                
                # Belleği temizle
                del batch_sequences, batch_targets, outputs
        
        self.train()
        return total_metric / num_batches
    
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
            # CPU'ya yükle
            checkpoint = torch.load(filename, map_location='cpu')
            
            # Checkpoint içeriğini kontrol et
            if 'model_state_dict' not in checkpoint:
                raise ValueError(f"Geçersiz model dosyası: 'model_state_dict' bulunamadı")
            
            # Parametre değerlerini kontrol et ve varsayılan değerler kullan
            hidden_size = checkpoint.get('hidden_size', 256)
            num_layers = checkpoint.get('num_layers', 3)
            input_size = checkpoint.get('input_size', 32)
            dropout_rate = float(checkpoint.get('dropout', 0.3))
            
            # Yeni bir model oluştur
            model = LSTMPredictor(
                config={
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout_rate,
                    'bidirectional': True
                }
            )
            
            # Model durumunu yükle
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"Model başarıyla yüklendi: {filename}")
            logger.info(f"Model başarıyla yüklendi: {filename}")
            return model
        
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {str(e)}")
            logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
            raise 