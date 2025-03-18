import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import os
import logging
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch.nn.utils.parametrizations  # parametrizations modülü
from config.config import MODEL_CONFIG  # MODEL_CONFIG'i import et
from utils.logger import log_epoch_results

logger = logging.getLogger("TradingBot.LSTMModel")

class LSTMPredictor(nn.Module):
    def __init__(self, config=None):
        """
        LSTM tabanlı fiyat tahmincisi başlatıcı
        
        Parametreler:
        - input_size: Giriş özellik sayısı
        - hidden_size: LSTM gizli katman boyutu
        - num_layers: LSTM katman sayısı
        - dropout: Dropout oranı
        """
        super(LSTMPredictor, self).__init__()
        
        # Konfigürasyonu kontrol et ve varsayılan değerlerle doldur
        if config is None:
            config = {}
        
        # Hyperparameters
        self.input_size = config.get('input_size', 32)       # Giriş özellik sayısı
        self.hidden_size = config.get('hidden_size', 128)    # LSTM gizli katman boyutu (artırıldı)
        self.num_layers = config.get('num_layers', 3)        # LSTM katman sayısı (artırıldı)
        self.dropout_rate = config.get('dropout', 0.3)       # Dropout oranı
        self.bidirectional = config.get('bidirectional', True)  # Çift yönlü LSTM
        
        # LeakyReLU için negatif slope değerini al
        self.leaky_relu_slope = config.get('leaky_relu_slope', 0.01)
        
        # LSTM çıktı boyutu (bidirectional ise 2*hidden_size)
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        # LSTM katmanı - Performans için daha karmaşık bir mimari
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            proj_size=0  # LSTM'in çıktı boyutunu sınırlamaz
        )
        
        # Dropout katmanları - Daha agresif dropout kullanımı
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.dropout3 = nn.Dropout(self.dropout_rate * 0.8)  # Son katmanda biraz daha az dropout
        
        # PReLU aktivasyon katmanları - Daha esnek aktivasyon
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        
        # Tam bağlantılı katmanlar - Daha geniş ve derin mimari
        fc1_size = 128  # İlk tam bağlantılı katman boyutu
        self.fc1 = nn.Linear(lstm_output_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, 1)
        
        # Batch Normalization - Daha stabil eğitim için
        self.bn1 = nn.BatchNorm1d(lstm_output_size)
        self.bn2 = nn.BatchNorm1d(fc1_size)
        
        # Modeli doğru cihaza taşı
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Model özet bilgilerini logla
        self._log_model_summary()
        
        # Parametreleri doğru şekilde başlat
        self._initialize_weights()
        
    def _log_model_summary(self):
        """
        Model yapılandırma bilgilerini loglar
        """
        logger.info(f"LSTM model oluşturuldu: input_size={self.input_size}, hidden_size={self.hidden_size}, " + 
                   f"num_layers={self.num_layers}, dropout={self.dropout_rate}, leaky_relu_slope={self.leaky_relu_slope}")
        logger.info(f"Model mimarisi: {'bidirectional' if self.bidirectional else 'unidirectional'} LSTM")
        
        # Parametre sayısını hesapla
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Toplam parametre sayısı: {total_params:,}, Eğitilebilir parametre sayısı: {trainable_params:,}")
        
        # Cihaz bilgisi
        logger.info(f"Model {self.device} üzerinde çalışacak")
        
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB cinsinden
            logger.info(f"GPU: {gpu_name}, Bellek: {gpu_memory:.2f} GB")
        
        # Layer Normalization (Batch Norm yerine)
        self.layer_norm = nn.LayerNorm(self.hidden_size * 2)
        
        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(
            self.hidden_size * 2,
            momentum=MODEL_CONFIG['batch_norm']['momentum'],
            eps=MODEL_CONFIG['batch_norm']['eps']
        )
        
        # Attention mekanizması
        attention_config = MODEL_CONFIG.get('attention', {
            'dims': [512, 128, 1],
            'dropout': 0.4             # Attention dropout oranı artırıldı (0.2'den 0.4'e)
        })
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size * 2, attention_config['dims'][0]),
            nn.LayerNorm(attention_config['dims'][0]),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            nn.Dropout(attention_config['dropout']),
            nn.Linear(attention_config['dims'][0], attention_config['dims'][1]),
            nn.Tanh(),
            nn.Linear(attention_config['dims'][1], attention_config['dims'][2])
        )
        
        # Fully connected katmanlar
        fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # Weight norm uygula (hem eski hem yeni API'yi destekler)
        try:
            # Yeni API
            self.fc1 = torch.nn.utils.parametrizations.weight_norm(fc1)
            logger.info("Yeni parametrizations.weight_norm API'si kullanılıyor")
        except AttributeError:
            # Eski API
            self.fc1 = torch.nn.utils.weight_norm(fc1)
            logger.info("Eski weight_norm API'si kullanılıyor")
        
        self.fc_norm = nn.LayerNorm(self.hidden_size)
        self.fc_dropout = nn.Dropout(self.dropout_rate)
        
        fc2 = nn.Linear(self.hidden_size, 1)
        
        # Weight norm uygula (hem eski hem yeni API'yi destekler)
        try:
            # Yeni API
            self.fc2 = torch.nn.utils.parametrizations.weight_norm(fc2)
        except AttributeError:
            # Eski API
            self.fc2 = torch.nn.utils.weight_norm(fc2)
        
        # Modeli seçilen cihaza taşı
        self.to(self.device)
        
        # Model oluşturuldu, şimdi ağırlıkları başlatalım
        try:
            self._initialize_parameters()
            logger.info("Model parametreleri başarıyla başlatıldı")
        except Exception as e:
            logger.error(f"Model parametreleri başlatılırken hata oluştu: {str(e)}")
            logger.warning("Model parametre başlatma hatası, varsayılan başlatma kullanılacak")
            # Hataya rağmen devam et
        
    def _initialize_parameters(self):
        """LSTM ve diğer katmanların ağırlıklarını başlatır"""
        try:
            for name, param in self.named_parameters():
                # Weight norm parametrelerini atla
                if 'weight_g' in name or 'weight_v' in name:
                    continue
                    
                if 'lstm' in name:
                    if 'weight' in name:
                        # LSTM ağırlıklarını Xavier ile başlat (ortogonal yerine)
                        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))
                    elif 'bias' in name:
                        # LSTM bias'larını sıfırla
                        nn.init.zeros_(param)
                elif ('fc' in name or 'attention' in name) and 'weight' in name:
                    # Sadece 2 veya daha fazla boyutu olan tensörler için Xavier kullan
                    if len(param.shape) >= 2:
                        # FC ve attention ağırlıklarını Xavier (Glorot) ile başlat
                        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('leaky_relu', self.leaky_relu_slope))
                    else:
                        # 1-D tensörler için normal dağılım kullan
                        logger.debug(f"1D tensor bulundu, normal başlatma kullanılıyor: {name}, shape: {param.shape}")
                        nn.init.normal_(param, mean=0.0, std=0.01)
                elif ('fc' in name or 'attention' in name) and 'bias' in name:
                    # Bias'ları sıfırla
                    nn.init.zeros_(param)
            
            logger.info("Model ağırlıkları başlatıldı: ortogonal (LSTM), xavier uniform (FC, Attention)")
        except Exception as e:
            logger.error(f"Parametre başlatılırken hata: {str(e)}")
            if 'name' in locals() and 'param' in locals():
                logger.error(f"Hataya neden olan parametre: {name}, shape: {param.shape if hasattr(param, 'shape') else 'bilinmiyor'}")
            raise
    
    def _initialize_weights(self):
        """
        _initialize_parameters metodunun alias'ı.
        Model parametrelerini başlatır.
        """
        return self._initialize_parameters()
    
    def calculate_accuracy(self, outputs, targets, threshold=0.005):
        """
        Tahminlerin doğruluk oranını hesaplar
        
        Parametreler:
        - outputs: Model çıktıları
        - targets: Gerçek hedef değerleri
        - threshold: Minimum değişim eşiği (çok küçük değerler için)
        
        Dönüş:
        - Doğruluk oranı (0.0-1.0 arası)
        """
        # Hedef değerlerin istatistiklerini kontrol et
        targets_mean = torch.mean(torch.abs(targets)).item()
        targets_std = torch.std(targets).item()
        
        # Dinamik threshold hesapla - minimum threshold değerini kullan
        adaptive_threshold = max(threshold, targets_mean * 0.05)
        
        # Log threshold değerini
        logger.debug(f"Accuracy hesaplaması - Threshold: {adaptive_threshold:.6f}, Targets Mean: {targets_mean:.6f}, Std: {targets_std:.6f}")
        
        # Önemli değişimleri belirle
        significant_targets = torch.abs(targets) > adaptive_threshold
        
        # Hiç önemli değişim yoksa veya çok az varsa
        if significant_targets.sum() < max(3, len(targets) * 0.1):  # En az 3 veya %10
            if len(targets) > 0:
                logger.warning(f"Çok az önemli değişim: {significant_targets.sum().item()}/{len(targets)}")
            else:
                logger.warning("Hedef değerler boş!")
            return 0.0
        
        # Önemli değişimler için doğruluk hesapla
        # Yön tahmini doğru mu?
        correct_direction = (torch.sign(outputs) == torch.sign(targets))
        
        # Önemli değişimlere ağırlık ver
        weight = 1 + torch.abs(targets) / (targets_std + 1e-8)  # Standart sapmaya göre ağırlıklandır
        
        # Ağırlıklı doğruluk hesapla
        accurate = ((correct_direction & significant_targets) * weight).sum()
        total_weight = (significant_targets * weight).sum()
        
        if total_weight > 0:
            accuracy = accurate / total_weight
            
            # Çok yüksek veya çok düşük doğruluk için uyarı
            if accuracy.item() > 0.9:
                logger.warning(f"Şüpheli yüksek doğruluk: %{accuracy.item()*100:.2f}")
            elif abs(accuracy.item() - 0.5) < 0.05:
                logger.warning(f"Doğruluk oranı rastgele tahmine yakın: %{accuracy.item()*100:.2f}")
            
            return accuracy.item()
        else:
            logger.warning("Ağırlıklandırılmış toplam sıfır, doğruluk hesaplanamıyor")
            return 0.0
    
    def _check_and_convert_input(self, x):
        """
        Giriş verisini kontrol et ve torch.Tensor'e dönüştür
        """
        if x is None:
            raise ValueError("Giriş verisi None olamaz")
            
        # Eğer torch.Tensor değilse dönüştürmeyi dene
        if not isinstance(x, torch.Tensor):
            try:
                logger.info(f"Giriş verisi torch.Tensor değil, dönüştürülüyor. Tip: {type(x)}")
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float()
                else:
                    x = torch.tensor(x, dtype=torch.float32)
                
                logger.info(f"Veri başarıyla dönüştürüldü. Yeni boyut: {x.shape}")
            except Exception as e:
                logger.error(f"Tensor dönüşümü hatası: {str(e)}")
                raise TypeError(f"Giriş torch.Tensor'e dönüştürülemedi: {type(x)}")
        
        # Tensörü doğru cihaza taşı
        if x.device != self.device:
            x = x.to(self.device)
            
        return x

    def forward(self, x):
        """
        İleri yayılım
        
        Parametreler:
        - x: Giriş tensörü
        
        Dönüş:
        - Çıkış tensörü
        """
        # Girişi kontrol et ve dönüştür
        x = self._check_and_convert_input(x)
        
        # LSTM katmanından geçir
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                         x.size(0), self.hidden_size, device=x.device)
        
        # Dropout uygula
        x = self.dropout1(x)
        
        # LSTM katmanından geçir
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Son zaman adımını al
        last_timestep = lstm_out[:, -1, :]
        
        # Batch normalizasyon (GPU için optimize)
        if last_timestep.size(0) > 1:  # Batch size 1'den büyükse
            last_timestep = self.bn1(last_timestep)
        
        # PReLU aktivasyonu
        out = self.prelu1(last_timestep)
        
        # Dropout
        out = self.dropout2(out)
        
        # Tam bağlantılı katman
        out = self.fc1(out)
        
        # Batch normalizasyon (GPU için optimize)
        if out.size(0) > 1:  # Batch size 1'den büyükse
            out = self.bn2(out)
        
        # PReLU aktivasyonu
        out = self.prelu2(out)
        
        # Son dropout ve tam bağlantılı katman
        out = self.dropout3(out)
        out = self.fc2(out)
        
        return out
    
    def train_model(self, train_sequences, train_targets, val_sequences=None, val_targets=None,
                   sample_weights=None, epochs=100, batch_size=32, learning_rate=0.001, patience=10, verbose=True,
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
        # Modeli eğitim moduna al - Bu çağrı eksikti ve hata buradan kaynaklanıyordu
        self.train()
        
        # Veri doğrulama
        if train_sequences is None:
            raise ValueError("train_sequences None olamaz")
            
        if train_targets is None:
            raise ValueError("train_targets None olamaz")
            
        # Eğer giriş numpy dizisi ise torch.Tensor'e dönüştür
        if not isinstance(train_sequences, torch.Tensor):
            train_sequences = torch.tensor(train_sequences, dtype=torch.float32)
        if not isinstance(train_targets, torch.Tensor):
            train_targets = torch.tensor(train_targets, dtype=torch.float32)
            
        # Tensörlerin boyutlarını kontrol et
        if len(train_sequences.shape) != 3:
            raise ValueError(f"train_sequences 3 boyutlu olmalıdır [batch, seq_len, features], alınan: {train_sequences.shape}")
            
        if len(train_targets.shape) == 1:
            train_targets = train_targets.view(-1, 1)
        elif len(train_targets.shape) != 2:
            raise ValueError(f"train_targets 1 veya 2 boyutlu olmalıdır, alınan: {train_targets.shape}")
            
        # Doğrulama verileri opsiyonel
        if val_sequences is not None and val_targets is not None:
            try:
                # Eğer giriş numpy dizisi ise torch.Tensor'e dönüştür
                if not isinstance(val_sequences, torch.Tensor):
                    val_sequences = torch.tensor(val_sequences, dtype=torch.float32)
                if not isinstance(val_targets, torch.Tensor):
                    val_targets = torch.tensor(val_targets, dtype=torch.float32)
                    
                # Tensörlerin boyutlarını kontrol et
                if len(val_sequences.shape) != 3:
                    logger.warning(f"val_sequences 3 boyutlu olmalıdır. Geçerli boyut: {val_sequences.shape}")
                    logger.warning("Doğrulama verileri atlanıyor...")
                    val_sequences = None
                    val_targets = None
            except Exception as e:
                logger.error(f"Doğrulama verileri dönüştürme hatası: {str(e)}")
                val_sequences = None
                val_targets = None
                
            # Boyut düzeltme
            if val_targets is not None and len(val_targets.shape) == 1:
                val_targets = val_targets.view(-1, 1)
        
        # Verileri GPU'ya taşı
        train_sequences = train_sequences.to(self.device)
        train_targets = train_targets.to(self.device)
        if val_sequences is not None:
            val_sequences = val_sequences.to(self.device)
            val_targets = val_targets.to(self.device)
        if sample_weights is not None:
            sample_weights = sample_weights.to(self.device)
        
        # Veri istatistiklerini logla
        logger.info("--- Eğitim Veri İstatistikleri ---")
        logger.info(f"Train sequences şekli: {train_sequences.shape}")
        logger.info(f"Train targets şekli: {train_targets.shape}")
        logger.info(f"Sequence değer aralığı: Min={train_sequences.min().item():.6f}, Max={train_sequences.max().item():.6f}")
        logger.info(f"Target değer aralığı: Min={train_targets.min().item():.6f}, Max={train_targets.max().item():.6f}")
        
        if torch.abs(train_targets).max().item() > 0.1:
            logger.warning(f"!!! UYARI: Hedef değerler -0.1 ile 0.1 aralığının dışında. "
                          f"Maksimum mutlak değer: {torch.abs(train_targets).max().item():.6f}")
        
        if val_sequences is not None:
            logger.info(f"Validation sequences şekli: {val_sequences.shape}")
            logger.info(f"Validation targets şekli: {val_targets.shape}")
            logger.info(f"Validation target değer aralığı: Min={val_targets.min().item():.6f}, "
                       f"Max={val_targets.max().item():.6f}")
        
        # Eğitim moduna geç
        self.train()
        
        # Ağırlıkları kontrol et
        if sample_weights is not None:
            logger.debug(f"Ağırlıklı eğitim kullanılıyor. Ağırlık aralığı: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
        
        # Öğrenme oranını kontrol et
        config_lr = MODEL_CONFIG['training'].get('learning_rate', 0.001)
        logger.info(f"Öğrenme oranı kontrolü - Fonksiyona gelen: {learning_rate}, Config dosyasındaki: {config_lr}")
        if learning_rate != config_lr:
            logger.warning(f"!!! DİKKAT: Kullanılan öğrenme oranı ({learning_rate}) config dosyasındaki değerden ({config_lr}) farklı!")
            learning_rate = config_lr
            logger.info(f"Öğrenme oranı config dosyasındaki değere ({config_lr}) güncellendi.")
        
        # Öğrenme oranının beklenmeyen değerleri için kontrol
        if learning_rate > 0.1 or learning_rate < 1e-6:
            logger.warning(f"!!! DİKKAT: Öğrenme oranı ({learning_rate}) olağandışı bir değer.")
            logger.info(f"Öğrenme oranı 0.0001 olarak ayarlanıyor.")
            learning_rate = 0.0001
        
        # Kayıp fonksiyonu ve metrikler
        def custom_loss(outputs, targets, weights=None):
            """Özel kayıp fonksiyonu - MSE ve yön doğruluğu kombinasyonu"""
            # Değerleri kontrol et
            if verbose and torch.isnan(outputs).any() or torch.isnan(targets).any():
                logger.warning("!!! NaN değerler bulundu! outputs veya targets içinde NaN var.")
                
            if verbose and torch.isinf(outputs).any() or torch.isinf(targets).any():
                logger.warning("!!! Inf değerler bulundu! outputs veya targets içinde Inf var.")
            
            # MSE loss
            mse_loss = F.mse_loss(outputs, targets, reduction='none')
            
            # Yön doğruluğu loss
            direction_correct = torch.sign(outputs) == torch.sign(targets)
            direction_loss = 1.0 - direction_correct.float()
            
            # Toplam kayıp
            loss_scale = MODEL_CONFIG['training'].get('loss_scale', 1.0)
            total_loss = (mse_loss + direction_loss * 0.1) * loss_scale  # Yön kaybına %10 ağırlık ver
            
            if weights is not None:
                total_loss = total_loss * weights.unsqueeze(1)
            
            mean_loss = total_loss.mean()
            
            # Çok büyük kayıp değerlerini kontrol et
            if mean_loss > 1000:
                logger.warning(f"!!! Çok yüksek kayıp değeri: {mean_loss.item():.6f}")
                logger.debug(f"MSE loss: {mse_loss.mean().item():.6f}, Direction loss: {direction_loss.mean().item():.6f}")
                logger.debug(f"Outputs aralığı: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
                logger.debug(f"Targets aralığı: [{targets.min().item():.6f}, {targets.max().item():.6f}]")
            
            return mean_loss
        
        # Optimizer ve scheduler
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=learning_rate,
            weight_decay=0.001
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=MODEL_CONFIG['training'].get('reduce_lr_factor', 0.5),
            patience=MODEL_CONFIG['training'].get('reduce_lr_patience', 7)
        )
        
        # Gradient clipping değerini al
        gradient_clip_value = MODEL_CONFIG['lstm'].get('gradient_clip', 1.0)
        logger.info(f"Gradient clipping değeri: {gradient_clip_value}")
        
        # Eğitim geçmişi
        train_losses = []
        val_losses = []
        # Doğruluk listeleri
        train_accuracies = []
        val_accuracies = []
        
        # Erken durdurma için değişkenler
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Gradyan akümülasyonu için değişkenler
        accumulation_steps = 4  # 4 batch biriktir
        optimizer.zero_grad()  # Optimizer gradyanları sıfırla
        
        try:
            # Eğitim döngüsü
            print(f"Eğitim başlıyor! {epochs} epoch, batch size: {batch_size}, learning rate: {learning_rate}")
            logger.info(f"Eğitim başlıyor! {epochs} epoch, batch size: {batch_size}, learning rate: {learning_rate}, accumulation_steps: {accumulation_steps}")
            
            # Başlangıç öğrenme oranını kaydet ve logla
            initial_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Başlangıç öğrenme oranı (optimizer'dan): {initial_lr}")
            
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
                batch_count = 0  # Akümülasyon sayacı
                
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
                    
                    # Accumulation için loss ölçekle
                    loss = loss / accumulation_steps
                    
                    # Accuracy hesapla
                    batch_acc = self.calculate_accuracy(outputs, batch_targets)
                    total_acc += batch_acc
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradyan akümülasyonu
                    batch_count += 1
                    if batch_count % accumulation_steps == 0 or i + batch_size >= train_sequences.shape[0]:
                        # Gradient değerlerini kontrol et
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=gradient_clip_value)
                        if torch.isnan(grad_norm):
                            raise ValueError("Gradient değerleri NaN")
                        
                        # Ayrıntılı gradyan değeri kontrolü
                        if epoch % 5 == 0 and i == 0:  # Her 5 epoch'ta bir, ilk batch'te kontrol et
                            grad_stats = {}
                            for name, param in self.named_parameters():
                                if param.grad is not None:
                                    # Güvenli std hesaplama
                                    try:
                                        std_val = param.grad.std().item() if param.grad.numel() > 1 else 0.0
                                    except:
                                        std_val = 0.0
                                    
                                    grad_stats[name] = {
                                        'min': param.grad.min().item(),
                                        'max': param.grad.max().item(),
                                        'mean': param.grad.mean().item(),
                                        'std': std_val
                                    }
                                    
                                    # Çok küçük veya çok büyük gradyanları tespit et
                                    if abs(param.grad.mean().item()) < 1e-8:
                                        logger.warning(f"!!! GRADİENT VANİSHİNG TESPİT EDİLDİ: {name} için gradyan neredeyse 0")
                                    elif abs(param.grad.mean().item()) > 10:
                                        logger.warning(f"!!! GRADİENT EXPLODING TESPİT EDİLDİ: {name} için gradyan çok büyük")
                            
                            logger.debug(f"Gradyan istatistikleri (Epoch {epoch+1}): {grad_stats}")
                        
                        # Optimizer step
                        optimizer.step()
                        optimizer.zero_grad()  # Optimizer gradyanları sıfırla
                    
                    # Toplam kayıp ve batch sayısını güncelle
                    total_loss += loss.item() * accumulation_steps
                    num_batches += 1
                
                # Ortalama metrikler
                avg_train_loss = total_loss / num_batches
                avg_train_acc = total_acc / num_batches
                train_losses.append(avg_train_loss)
                train_accuracies.append(avg_train_acc)  # Eğitim doğruluğunu kaydet
                
                # Aşırı öğrenme kontrolü
                if avg_train_acc > 0.75 and epoch > 10:
                    logger.warning(f"!!! AŞIRI ÖĞRENME TESPİT EDİLDİ: Eğitim doğruluğu çok yüksek (%{avg_train_acc*100:.2f}).")
                    logger.warning(f"Model eğitim verisini ezberlemiş olabilir. Regularizasyon artırılmalı veya model karmaşıklığı azaltılmalı.")
                    
                    # Aşırı öğrenme devam ederse erken durdurma
                    if avg_train_acc > 0.85 and avg_train_loss < 0.01:
                        logger.warning("Aşırı öğrenme nedeniyle eğitim erken sonlandırılıyor.")
                        if verbose:
                            print("\nAşırı öğrenme nedeniyle eğitim erken sonlandırılıyor.")
                        break
                
                # Doğrulama
                val_loss = None
                val_acc = None
                
                if val_sequences is not None and val_targets is not None:
                    val_loss = self._validate(val_sequences, val_targets)
                    val_acc = self._validate(val_sequences, val_targets, metric='accuracy')
                    val_losses.append(val_loss)
                    val_accuracies.append(val_acc)  # Doğrulama doğruluğunu kaydet
                    
                    # Scheduler'ı güncelle
                    scheduler.step(val_loss)
                    
                    # Progress bar'ı güncelle
                    pbar.set_postfix({
                        'train_loss': f'{avg_train_loss:.6f}',
                        'train_acc': f'{avg_train_acc:.2%}',
                        'val_loss': f'[VAL] {val_loss:.6f}',  # [VAL] ön eki ekleyerek daha belirgin yaptım
                        'val_acc': f'[VAL] {val_acc:.2%}',    # [VAL] ön eki ekleyerek daha belirgin yaptım
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    
                    # Terminalde validation metriklerini ayrıca göster
                    if verbose and (epoch % 5 == 0 or epoch == epochs - 1):  # Her 5 epoch'ta bir veya son epoch'ta
                        print(f"\nEpoch {epoch+1}/{epochs}")
                        print(f"  - Train Loss: {avg_train_loss:.6f}, Train Accuracy: {avg_train_acc:.2%}")
                        print(f"  - Val Loss: {val_loss:.6f}, Val Accuracy: {val_acc:.2%}")
                    
                    # Her epoch'un sonuçlarını logla
                    current_lr = optimizer.param_groups[0]["lr"]
                    log_epoch_results(
                        epoch=epoch,
                        epochs=epochs,
                        train_loss=avg_train_loss,
                        train_acc=avg_train_acc,
                        val_loss=val_loss,
                        val_acc=val_acc,
                        lr=current_lr
                    )
                    
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
                    
                    # Terminalde training metriklerini ayrıca göster
                    if verbose and (epoch % 5 == 0 or epoch == epochs - 1):  # Her 5 epoch'ta bir veya son epoch'ta
                        print(f"\nEpoch {epoch+1}/{epochs}")
                        print(f"  - Train Loss: {avg_train_loss:.6f}, Train Accuracy: {avg_train_acc:.2%}")
                        print(f"  - Validation verisi kullanılmıyor")
                    
                    # Her epoch'un sonuçlarını logla (validation olmadan)
                    current_lr = optimizer.param_groups[0]["lr"]
                    log_epoch_results(
                        epoch=epoch,
                        epochs=epochs,
                        train_loss=avg_train_loss,
                        train_acc=avg_train_acc,
                        lr=current_lr
                    )
                    
                    # Save checkpoint if specified
                    if checkpoint_save and epoch > 0 and (epoch + 1) % checkpoint_interval == 0:
                        checkpoint_save(self.state_dict())
            
            # Son durumu logla
            final_message = f"Eğitim tamamlandı. Son Eğitim Kaybı: {train_losses[-1]:.6f}, Son Eğitim Doğruluğu: %{train_accuracies[-1]*100:.2f}"
            if val_losses:
                final_message += f", Son Doğrulama Kaybı: {val_losses[-1]:.6f}, Son Doğrulama Doğruluğu: %{val_accuracies[-1]*100:.2f}"
            logger.info(final_message)
            if verbose:
                print(f"\n{final_message}")
            
            return {
                'train_losses': train_losses, 
                'val_losses': val_losses if val_sequences is not None else None,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies if val_sequences is not None else None
            }
            
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
        # Validation için modeli eval moduna al
        self.eval()
        total_metric = 0
        num_batches = 0
        criterion = nn.MSELoss(reduction='mean')
        
        try:
            # Validation başlamadan önce log
            logger.info(f"Validation başlıyor. Veri boyutu: {val_sequences.shape}")
            
            with torch.no_grad():
                for i in range(0, len(val_sequences), batch_size):
                    batch_sequences = val_sequences[i:i+batch_size]
                    batch_targets = val_targets[i:i+batch_size]
                    
                    outputs = self(batch_sequences)
                    
                    if metric == 'loss':
                        batch_metric = criterion(outputs, batch_targets).item()
                    elif metric == 'accuracy':
                        batch_metric = self.calculate_accuracy(outputs, batch_targets)
                    else:
                        raise ValueError(f"Bilinmeyen metrik: {metric}")
                    
                    total_metric += batch_metric
                    num_batches += 1
            
            # Sonucu hesapla
            avg_metric = total_metric / max(num_batches, 1)  # Sıfıra bölünmeyi önle
            
            # Sonucu logla
            logger.info(f"Validation tamamlandı. {metric}: {avg_metric:.6f}")
            
            # Validasyondan sonra modeli tekrar train moduna al
            self.train()
            
            return avg_metric
            
        except Exception as e:
            logger.error(f"Validation sırasında hata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Hata durumunda varsayılan değer dön
            if metric == 'loss':
                return float('inf')  # Kayıp için yüksek değer
            else:
                return 0.0  # Accuracy için 0
    
    def evaluate(self, sequences, targets):
        """
        Model performansını değerlendir
        
        Parametreler:
        - sequences: Değerlendirilecek veri dizileri
        - targets: Hedef değerler
        
        Dönüş:
        - Kayıp ve doğruluk
        """
        # Değerlendirme için eval moduna geç
        self.eval()
        
        try:
            with torch.no_grad():
                outputs = self(sequences)
                loss = nn.MSELoss()(outputs, targets).item()
                accuracy = self.calculate_accuracy(outputs, targets)
                
            logger.info(f"Model değerlendirmesi - Kayıp: {loss:.6f}, Doğruluk: %{accuracy*100:.2f}")
            return loss, accuracy
        
        finally:
            # İşlem sonunda modeli tekrar train moduna al
            self.train()
    
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
            'dropout': self.dropout_rate
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