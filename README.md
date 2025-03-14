# XAUUSD Trading Bot

> # ⚠️ UYARI: GELİŞTİRME AŞAMASI ⚠️
> ## Bu proje aktif geliştirme aşamasındadır!
> - Bu yazılım şu anda test ve geliştirme sürecindedir
> - Üretim ortamında kullanım için henüz hazır değildir
> - Kullanımdan doğabilecek riskler kullanıcıya aittir

Bu proje, XAUUSD (Altın) paritesi için makine öğrenimini kullanan otomatik ticaret botudur. Bot, LSTM ve Pekiştirmeli Öğrenme (RL) modellerini kullanarak ticaret kararları verir.

## 🚀 Özellikler

- **Çoklu Zaman Dilimi Analizi**: 5m, 15m ve 1h grafiklerde eşzamanlı analiz
- **Hibrit Yapay Zeka Modeli**: 
  - LSTM ve RL modellerinin sıralı entegrasyonu
    - LSTM modeli fiyat tahminleri yapar
    - RL modeli LSTM tahminlerini kullanarak işlem kararları verir
  - 32 farklı özellik kullanımı
  - Optimize edilmiş model parametreleri
  - Çift yönlü LSTM ve dikkat mekanizması
  - Batch normalization ve gelişmiş dropout
  - Otomatik CUDA/CPU optimizasyonu
  - **Google Colab Entegrasyonu**: 
    - Otomatik model eğitimi
    - Veri senkronizasyonu
    - Eğitim durumu takibi
    - Model indirme/yükleme
- **Gelişmiş Risk Yönetimi**: 
  - Her ticaret için maksimum %1 risk
  - Maksimum %5 günlük zarar limiti
  - Maksimum %15 toplam zarar limiti
  - ATR tabanlı dinamik trailing stop
  - Çoklu seviyeli kar alma stratejisi (1.5R, 2R, 3R)
- **MT5 Entegrasyonu**: 
  - Tam MetaTrader 5 entegrasyonu
  - Otomatik yeniden bağlanma
  - Güvenli bağlantı yönetimi
- **Sistem İzleme ve Güvenlik**:
  - Bellek ve CPU kullanımı optimizasyonu
  - Otomatik garbage collection
  - Hassas veri filtreleme
  - Detaylı loglama sistemi
  - Gradient değeri kontrolleri
  - Veri doğrulama ve hata yakalama
- **Bildirim Sistemi**:
  - Telegram entegrasyonu (isteğe bağlı)
  - Emoji ile zenginleştirilmiş durum mesajları
  - Kritik durum uyarıları
  - Model performans metrikleri

## 📋 Gereksinimler

- Python 3.8+
- MetaTrader 5
- CUDA uyumlu GPU (opsiyonel, performans için önerilir)
- Google hesabı (Colab entegrasyonu için)
- Gerekli Python kütüphaneleri (`requirements.txt` dosyasında listelenmiştir)

## 🛠️ Kurulum

1. Depoyu klonlayın:
```bash
git clone https://github.com/AhmetSahinCAKIR/XAUUSD.git
cd XAUUSD
```

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv .venv
.venv\Scripts\activate     # Windows için
source .venv/bin/activate  # Linux/Mac için
```

3. Gerekli kütüphaneleri kurun:
```bash
pip install -r requirements.txt
```

4. Yapılandırma dosyalarını hazırlayın:
```bash
copy .env.example .env    # Windows için
cp .env.example .env     # Linux/Mac için
```

5. `.env` dosyasını düzenleyin:
   - MT5_LOGIN: MetaTrader 5 hesap numaranız
   - MT5_PASSWORD: MetaTrader 5 şifreniz
   - MT5_SERVER: Broker sunucu adı
   - Diğer parametreleri isteğe bağlı olarak ayarlayın

6. Google Cloud Console'dan API credentials oluşturun:
   - Yeni bir proje oluşturun
   - Google Drive API'yi etkinleştirin
   - OAuth 2.0 credentials oluşturun
   - İndirilen credentials dosyasını `config/credentials.json` olarak kaydedin

7. Google Drive'da gerekli klasörleri oluşturun:
   - `trading_bot` ana klasörü
   - `models` alt klasörü (eğitilen modeller için)
   - `data` alt klasörü (eğitim verileri için)

8. `config/colab_config.json` dosyasını düzenleyin:
   - `colab_notebook_id`: Colab notebook ID'si
   - `drive_folders.models`: Models klasörü ID'si
   - `drive_folders.data`: Data klasörü ID'si

9. Gerekli dizinleri oluşturun:
```bash
mkdir -p logs data saved_models config notebooks
```

## 💻 Kullanım

1. MetaTrader 5'i başlatın ve hesabınıza giriş yapın.

2. Botu çalıştırın:
```bash
python main.py
```

Bot başlatıldığında:
- MT5 bağlantısını kontrol eder
- Sistem kaynaklarını izlemeye başlar
- GPU kullanılabilirliğini kontrol eder
- Modelleri yükler veya eğitir (Colab entegrasyonu ile)
- Gerçek zamanlı trading başlar

## 🤖 Model Eğitimi

Bot iki şekilde model eğitimi yapabilir:

1. **Yerel Eğitim**:
   - Düşük veri miktarı
   - Hızlı eğitim
   - Sistem kaynaklarını kullanır

2. **Google Colab Eğitimi** (Önerilen):
   - Yüksek veri miktarı
   - GPU hızlandırma
   - Sistem kaynaklarını kullanmaz
   - Otomatik senkronizasyon
   - İlerleme takibi

### 🚧 Colab Entegrasyonu - Yapılacaklar

> ⚠️ **NOT**: Google Colab entegrasyonu şu anda geliştirme aşamasındadır.

Tamamlanması gereken özellikler:
1. **Notebook Güncellemeleri**:
   - LSTM ve RL modellerinin sıralı entegrasyonu
   - LSTM tahminlerinin RL modeline aktarılması
   - Model performans metriklerinin genişletilmesi
   - Hyperparameter optimizasyonu desteği

2. **Veri İşleme İyileştirmeleri**:
   - Veri ön işleme pipeline'ının güncellenmesi
   - Feature engineering süreçlerinin otomatikleştirilmesi
   - Veri kalitesi kontrollerinin eklenmesi
   - Veri augmentasyon tekniklerinin uygulanması

3. **Model Eğitim Geliştirmeleri**:
   - Early stopping mekanizması
   - Model checkpoint sistemi
   - Cross-validation desteği
   - Ensemble learning teknikleri
   - Transfer learning desteği

4. **Entegrasyon İyileştirmeleri**:
   - Colab session yönetiminin geliştirilmesi
   - Otomatik notebook yükleme/güncelleme
   - Eğitim durumu izleme sisteminin genişletilmesi
   - Hata yakalama ve kurtarma mekanizmaları

5. **Belgelendirme ve Testler**:
   - Detaylı API dokümantasyonu
   - Örnek kullanım senaryoları
   - Unit test ve integration testleri
   - Performance benchmark testleri

Bu özellikler tamamlandığında:
- Daha stabil ve güvenilir model eğitimi
- Daha iyi performans metrikleri
- Daha kolay kullanım ve bakım
- Daha güvenli veri yönetimi
sağlanacaktır.

Colab eğitimi seçildiğinde:
1. MT5'ten veri çekilir
2. Veri Google Drive'a yüklenir
3. Colab'da eğitim başlatılır
4. Eğitim durumu izlenir
5. Model indirilir ve kullanıma hazır hale gelir

## ⚙️ Yapılandırma

Temel parametreler `.env` dosyasında ayarlanabilir:
- Risk yönetimi parametreleri
- Bağlantı bilgileri
- Bildirim ayarları

Gelişmiş parametreler:
- `config.py`: Model ve sistem parametreleri
- `config/colab_config.json`: Colab entegrasyon ayarları

## 📊 İzleme ve Raporlama

Bot çalışırken:
- Anlık durum bilgileri konsola yazdırılır
- Detaylı loglar `logs/` dizinine kaydedilir
- Sistem durumu sürekli izlenir
- Model performans metrikleri kaydedilir
- İsteğe bağlı Telegram bildirimleri gönderilir

Log dosyaları:
- `logs/trading_bot.log`: Genel işlem logları
- `logs/error.log`: Hata logları
- `logs/model_performance.log`: Model metrikleri

## 🔄 Otomatik Yeniden Başlatma

Bot şu durumlarda otomatik olarak yeniden bağlanır:
- MT5 bağlantısı koptuğunda
- Bellek kullanımı yükseldiğinde
- Kritik hatalar oluştuğunda
- NaN gradient değerleri tespit edildiğinde
- Veri doğrulama hataları oluştuğunda
- Colab bağlantısı kesildiğinde

## ⚠️ Risk Uyarısı

Bu bot deneyseldir ve finansal tavsiye teşkil etmez. Gerçek hesapta kullanmadan önce:
- Demo hesapta kapsamlı testler yapın
- Risk parametrelerini dikkatle ayarlayın
- Piyasa koşullarını sürekli izleyin
- Model performansını değerlendirin
- Bellek kullanımını takip edin

## 🔍 Hata Ayıklama

Sorun yaşarsanız:
1. Log dosyalarını kontrol edin
2. MT5 bağlantısını doğrulayın
3. `.env` dosyasındaki bilgileri kontrol edin
4. Sistem kaynaklarının yeterli olduğundan emin olun
5. Model performans metriklerini inceleyin
6. GPU kullanılabilirliğini kontrol edin
7. Bellek kullanımı istatistiklerini gözden geçirin
8. Google credentials'ın doğru olduğunu kontrol edin
9. Drive klasör izinlerini kontrol edin
