# XAUUSD Trading Bot

Bu proje, XAUUSD (Altın) paritesi için makine öğrenimini kullanan otomatik ticaret botudur. Bot, LSTM ve Pekiştirmeli Öğrenme (RL) modellerini kullanarak ticaret kararları verir.

## 🚀 Özellikler

- **Çoklu Zaman Dilimi Analizi**: 1m, 5m ve 15m grafiklerde eşzamanlı analiz
- **Hibrit Yapay Zeka Modeli**: LSTM ve RL modellerinin kombinasyonu
- **Otomatik Risk Yönetimi**: 
  - Her ticaret için maksimum %1 risk
  - Maksimum %5 günlük zarar limiti
  - ATR tabanlı dinamik stop loss
  - Risk/Ödül oranına dayalı kar alma seviyeleri
- **MT5 Entegrasyonu**: MetaTrader 5 ile tam entegrasyon

## 📋 Gereksinimler

- Python 3.8+
- MetaTrader 5
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

4. MetaTrader 5'i kurun ve demo hesabı oluşturun.

## 💻 Kullanım

1. MetaTrader 5'i başlatın ve hesabınıza giriş yapın.

2. config.py dosyasındaki MT5_CONFIG ayarlarını kendi hesap bilgilerinizle güncelleyin:
```python
MT5_CONFIG = {
    'login': HESAP_NUMARANIZ,  
    'password': 'ŞİFRENİZ',    
    'server': 'SUNUCU_ADINIZ'  
}
```

3. Botu çalıştırın:
```bash
python main.py
```

İlk başlatıldığında, bot şunları yapacaktır:
- Gerekli modelleri oluşturma
- Tarihsel verilerle modelleri eğitme
- Ardından gerçek zamanlı ticaret başlatma

## ⚙️ Yapılandırma

Temel parametreler `config.py` dosyasında ayarlanabilir:
- `initial_balance`: Başlangıç bakiyesi
- `risk_per_trade`: İşlem başına risk yüzdesi
- `max_daily_loss`: Maksimum günlük zarar yüzdesi
- `timeframes`: Analiz edilecek zaman dilimleri

## 📊 Performans İzleme

Çalışırken, bot şunları yazdırır:
- Mevcut bakiye
- Günlük kar/zarar
- Açılan işlemler
- Model tahminleri
ve diğer bilgiler konsola.

## 🔍 Hata Giderme

Bot çalışmazsa şunları kontrol edin:
1. MetaTrader 5 terminalinin açık olduğundan emin olun
2. Doğru hesap bilgilerinin `config.py` dosyasında olduğunu doğrulayın
3. Piyasa saatlerinde çalıştırdığınızdan emin olun (hafta sonu çalışmaz)
4. MT5 terminalinde "Araçlar > Seçenekler > Uzman Danışmanlar" menüsünden API izinlerini etkinleştirin

## ⚠️ Risk Uyarısı

Bu bot deneyseldir ve finansal tavsiye teşkil etmez. Gerçek hesapta kullanmadan önce:
- Bir demo hesabında kapsamlı bir şekilde test edin
- Risk yönetimi parametrelerini dikkatle ayarlayın
- Piyasa koşullarını sürekli olarak izleyin

## 📝 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın. 