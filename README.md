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
  - LSTM ve RL modellerinin kombinasyonu
  - 32 farklı özellik kullanımı
  - Optimize edilmiş model parametreleri
  - Çift yönlü LSTM ve dikkat mekanizması
  - Batch normalization ve gelişmiş dropout
  - Otomatik CUDA/CPU optimizasyonu
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

4. Yapılandırma dosyasını hazırlayın:
```bash
copy .env.example .env    # Windows için
cp .env.example .env     # Linux/Mac için
```

5. `.env` dosyasını düzenleyin:
   - MT5_LOGIN: MetaTrader 5 hesap numaranız
   - MT5_PASSWORD: MetaTrader 5 şifreniz
   - MT5_SERVER: Broker sunucu adı
   - Diğer parametreleri isteğe bağlı olarak ayarlayın

6. Gerekli dizinleri oluşturun:
```bash
mkdir -p logs data saved_models
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
- Modelleri yükler veya eğitir
- Gerçek zamanlı trading başlar

## ⚙️ Yapılandırma

Temel parametreler `.env` dosyasında ayarlanabilir:
- Risk yönetimi parametreleri
- Bağlantı bilgileri
- Bildirim ayarları

Gelişmiş parametreler `config.py` dosyasında bulunur:
- Model parametreleri (LSTM ve RL)
- Trading stratejisi ayarları
- Sistem yapılandırması
- Loglama ayarları
- Bellek yönetimi limitleri
- Batch boyutları ve optimizasyon ayarları

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

## 📝 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın. 