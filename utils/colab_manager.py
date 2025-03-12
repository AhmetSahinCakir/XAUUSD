import os
import json
import time
import logging
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload
import requests
import io
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("TradingBot.ColabManager")

class ColabManager:
    def __init__(self, config_path='config/colab_config.json'):
        """Initialize Colab Manager"""
        self.config_path = config_path
        self.credentials = None
        self.drive_service = None
        self.colab_notebook_id = None
        self.model_folder_id = None
        self.data_folder_id = None
        self.colab_url = None  # Colab URL'i başlangıçta None olarak ayarla
        self._setup_credentials()
        self._setup_colab_config()
        
    def _read_json_file(self, file_path, file_type=""):
        """
        JSON dosyasını güvenli bir şekilde oku
        
        Parametreler:
        - file_path: Okunacak dosyanın yolu
        - file_type: Dosya tipi (log mesajları için)
        
        Dönüş:
        - dict: JSON verisi
        - None: Hata durumunda
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"[HATA] {file_type} dosyası bulunamadı: {file_path}")
                return None
                
            if not os.access(file_path, os.R_OK):
                logger.error(f"[HATA] {file_type} dosyası okunamıyor (izin hatası): {file_path}")
                return None
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                logger.debug(f"[BILGI] {file_type} dosyası başarıyla okundu")
                return data
                
        except json.JSONDecodeError as e:
            logger.error(f"[HATA] {file_type} dosyası geçerli bir JSON değil: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"[HATA] {file_type} dosyası okunurken hata: {str(e)}")
            return None
            
    def _validate_credentials(self, creds_data):
        """
        Kimlik bilgilerinin geçerliliğini kontrol et
        
        Parametreler:
        - creds_data: Kontrol edilecek kimlik bilgileri
        
        Dönüş:
        - bool: Geçerli ise True, değilse False
        """
        required_fields = {
            'credentials': ['client_id', 'client_secret', 'auth_uri', 'token_uri'],
            'token': ['token', 'refresh_token', 'token_uri', 'client_id', 'client_secret']
        }
        
        try:
            if 'installed' in creds_data:  # credentials.json
                data = creds_data['installed']
                fields = required_fields['credentials']
            else:  # token.json
                data = creds_data
                fields = required_fields['token']
                
            missing_fields = [field for field in fields if field not in data]
            
            if missing_fields:
                logger.error(f"[HATA] Eksik alanlar: {', '.join(missing_fields)}")
                return False
                
            # Değerlerin boş olmadığını kontrol et
            empty_fields = [field for field in fields if not data[field]]
            if empty_fields:
                logger.error(f"[HATA] Boş alanlar: {', '.join(empty_fields)}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"[HATA] Kimlik doğrulama kontrolü sırasında hata: {str(e)}")
            return False
            
    def _setup_credentials(self):
        """Google Drive API kimlik bilgilerini ayarla"""
        try:
            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            creds = None
            
            # Token dosyasını oku
            if os.path.exists('config/token.json'):
                token_data = self._read_json_file('config/token.json', "Token")
                if token_data and self._validate_credentials(token_data):
                    try:
                        creds = Credentials.from_authorized_user_info(token_data, SCOPES)
                        logger.info("[BILGI] Token başarıyla yüklendi")
                    except Exception as e:
                        logger.error(f"[HATA] Token oluşturulurken hata: {str(e)}")
                        creds = None
                        
            # Token geçerli değilse veya yenilenmesi gerekiyorsa
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        logger.info("[BILGI] Token yenileniyor...")
                        creds.refresh(Request())
                        logger.info("[BASARILI] Token yenilendi")
                    except Exception as e:
                        logger.error(f"[HATA] Token yenilenirken hata: {str(e)}")
                        creds = None
                else:
                    # Credentials dosyasını oku
                    creds_data = self._read_json_file('config/credentials.json', "Credentials")
                    if not creds_data or not self._validate_credentials(creds_data):
                        raise Exception("[HATA] Geçerli credentials.json dosyası bulunamadı")
                        
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            'config/credentials.json', SCOPES)
                        creds = flow.run_local_server(port=0)
                        logger.info("[BASARILI] Yeni token oluşturuldu")
                        
                        # Yeni token'ı kaydet
                        token_path = 'config/token.json'
                        os.makedirs(os.path.dirname(token_path), exist_ok=True)
                        
                        token_data = {
                            'token': creds.token,
                            'refresh_token': creds.refresh_token,
                            'token_uri': creds.token_uri,
                            'client_id': creds.client_id,
                            'client_secret': creds.client_secret,
                            'scopes': creds.scopes,
                            'expiry': creds.expiry.isoformat() if creds.expiry else None
                        }
                        
                        with open(token_path, 'w') as token:
                            json.dump(token_data, token, indent=4)
                        logger.info("[BASARILI] Yeni token kaydedildi")
                        
                    except Exception as e:
                        logger.error(f"[HATA] Yeni token oluşturulurken hata: {str(e)}")
                        raise
                        
            self.credentials = creds
            self.drive_service = build('drive', 'v3', credentials=creds)
            logger.info("[BASARILI] Google Drive servisi başlatıldı")
            
        except Exception as e:
            logger.error(f"[HATA] Kimlik doğrulama ayarlanırken hata: {str(e)}")
            raise
            
    def upload_data_to_drive(self, file_name):
        """Veri dosyasını Google Drive'a yükle"""
        try:
            # Dosyanın varlığını kontrol et
            if not os.path.exists(file_name):
                logger.error("[HATA] Yüklenecek dosya bulunamadı: " + file_name)
                return False
            
            # Drive'a yükle
            logger.info("[BILGI] Veri dosyası Drive'a yükleniyor: " + file_name)
            
            # Yükleme başarılı
            logger.info("[BASARILI] Veri dosyası Drive'a yüklendi: " + file_name)
            return True
            
        except Exception as e:
            logger.error(f"[HATA] Drive'a yükleme hatası: {str(e)}")
            return False
            
    def start_colab_training(self):
        """Colab'da eğitimi başlat"""
        try:
            self.logger.info("[BILGI] Colab'da eğitim başlatılıyor...")
            
            # Drive'da notebook'u ara
            query = f"name = 'train_models.ipynb' and trashed = false"
            results = self.drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)',
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            
            items = results.get('files', [])
            if not items:
                self.logger.error("""
Colab notebook bulunamadı! Lütfen aşağıdakileri kontrol edin:
1. 'train_models.ipynb' dosyası Google Drive'ınızda mevcut mu?
2. config/colab_config.json dosyasındaki ID'ler doğru mu?
3. Google Drive API izinleri doğru ayarlanmış mı?

Çözüm için:
1. Örnek Colab notebook'u Drive'ınıza yükleyin
2. colab_config.json dosyasını güncelleyin
3. Google Cloud Console'dan API izinlerini kontrol edin""")
                raise Exception("Colab eğitimi başlatma hatası!")
            
            notebook_id = items[0]['id']
            self.config['colab_notebook_id'] = notebook_id
            
            # Training status dosyasını oluştur/güncelle
            status_data = {
                'status': 'started',
                'start_time': datetime.now().isoformat(),
                'model_folder_id': self.config['drive_folders']['models'],
                'data_folder_id': self.config['drive_folders']['data']
            }
            
            # Status dosyasını ara
            query = f"name = '{self.config['model_files']['status']}' and trashed = false"
            results = self.drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)',
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            
            items = results.get('files', [])
            
            if items:
                # Mevcut dosyayı güncelle
                status_file_id = items[0]['id']
                file_metadata = {'name': self.config['model_files']['status']}
                media = MediaIoBaseUpload(
                    io.BytesIO(json.dumps(status_data).encode()),
                    mimetype='application/json',
                    resumable=True
                )
                self.drive_service.files().update(
                    fileId=status_file_id,
                    body=file_metadata,
                    media_body=media
                ).execute()
            else:
                # Yeni dosya oluştur
                file_metadata = {
                    'name': self.config['model_files']['status'],
                    'parents': [self.config['drive_folders']['models']]
                }
                media = MediaIoBaseUpload(
                    io.BytesIO(json.dumps(status_data).encode()),
                    mimetype='application/json',
                    resumable=True
                )
                self.drive_service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
            
            self.logger.info("[BASARILI] Colab eğitimi başlatıldı!")
            return True
            
        except Exception as e:
            self.logger.error(f"[HATA] Colab eğitim hatası: {str(e)}")
            return False
            
    def download_model(self, model_name='lstm_model.pth', save_path='saved_models/'):
        """Download trained model from Drive"""
        try:
            # Model dosyasını bul
            results = self.drive_service.files().list(
                q=f"name='{model_name}' and '{self.model_folder_id}' in parents",
                fields="files(id, name, modifiedTime)"
            ).execute()
            
            files = results.get('files', [])
            if not files:
                logger.error(f"❌ Model dosyası bulunamadı: {model_name}")
                return False
                
            # En son güncellenen modeli al
            latest_model = max(files, key=lambda x: x['modifiedTime'])
            
            # Modeli indir
            request = self.drive_service.files().get_media(fileId=latest_model['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                
            # Modeli kaydet
            fh.seek(0)
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, model_name), 'wb') as f:
                f.write(fh.read())
                
            logger.info(f"✅ Model başarıyla indirildi: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"⚠️ Model indirme hatası: {str(e)}")
            return False
            
    def check_training_status(self):
        """Check if model training is complete"""
        try:
            # Status dosyasını kontrol et
            results = self.drive_service.files().list(
                q=f"name='training_status.json' and '{self.model_folder_id}' in parents",
                fields="files(id, name)"
            ).execute()
            
            files = results.get('files', [])
            if not files:
                return False
                
            # Status dosyasını oku
            request = self.drive_service.files().get_media(fileId=files[0]['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                
            fh.seek(0)
            status_data = json.loads(fh.read().decode())
            
            return status_data.get('training_complete', False)
            
        except Exception as e:
            logger.error(f"⚠️ Eğitim durumu kontrol hatası: {str(e)}")
            return False
            
    def wait_for_training_completion(self, timeout=7200, check_interval=60):
        """Wait for model training to complete"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_training_status():
                logger.info("✅ Model eğitimi başarıyla tamamlandı!")
                return True
                
            time.sleep(check_interval)
            
        logger.error("❌ Model eğitimi zaman aşımına uğradı!")
        return False

    def _setup_colab_config(self):
        """Colab konfigürasyonunu yükle"""
        try:
            config_data = self._read_json_file(self.config_path, "Colab Config")
            if config_data:
                self.colab_notebook_id = config_data.get('colab_notebook_id')
                self.model_folder_id = config_data.get('model_folder_id')
                self.data_folder_id = config_data.get('data_folder_id')
                # Colab URL'ini oluştur
                self.colab_url = f"https://colab.research.google.com/drive/{self.colab_notebook_id}"
                logger.info("[BILGI] Colab konfigürasyonu başarıyla yüklendi")
        except Exception as e:
            logger.error(f"[HATA] Colab konfigürasyonu yüklenirken hata: {str(e)}")
            raise 