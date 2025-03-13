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
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_dir = os.path.join(self.base_dir, 'config')
        self.config_path = os.path.join(self.base_dir, config_path)
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
            token_path = os.path.join(self.config_dir, 'token.json')
            if os.path.exists(token_path):
                token_data = self._read_json_file(token_path, "Token")
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
                    credentials_path = os.path.join(self.config_dir, 'credentials.json')
                    creds_data = self._read_json_file(credentials_path, "Credentials")
                    if not creds_data or not self._validate_credentials(creds_data):
                        raise Exception("[HATA] Geçerli credentials.json dosyası bulunamadı")
                        
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            credentials_path, SCOPES)
                        creds = flow.run_local_server(port=0)
                        logger.info("[BASARILI] Yeni token oluşturuldu")
                        
                        # Yeni token'ı kaydet
                        token_path = os.path.join(self.config_dir, 'token.json')
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
            
            file_metadata = {
                'name': os.path.basename(file_name),
                'parents': [self.config['drive_folders']['data']]
            }
            
            media = MediaFileUpload(file_name, resumable=True)
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            # Dosyanın başarıyla yüklenip yüklenmediğini kontrol et
            uploaded_file_id = file.get('id')
            if not uploaded_file_id:
                logger.error("[HATA] Dosya yüklendi ancak ID alınamadı")
                return False
                
            # Drive'da dosyanın varlığını kontrol et
            try:
                self.drive_service.files().get(fileId=uploaded_file_id).execute()
                logger.info("[BASARILI] Veri dosyası Drive'a yüklendi ve doğrulandı: " + file_name)
                return True
            except Exception as e:
                logger.error(f"[HATA] Dosya yüklendi ancak doğrulanamadı: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"[HATA] Drive'a yükleme hatası: {str(e)}")
            return False
            
    def start_colab_training(self):
        """Colab'da eğitimi başlat"""
        try:
            logger.info("[BILGI] Colab'da eğitim başlatılıyor...")
            
            # Trading bot klasörünü ara
            trading_bot_folder_id = self.config['drive_folders'].get('trading_bot')
            if not trading_bot_folder_id:
                logger.error("Trading bot klasör ID'si bulunamadı!")
                raise Exception("Trading bot klasör ID'si eksik!")

            # Drive'da notebook'u ara
            query = f"name = '{self.config['notebook']['name']}' and '{trading_bot_folder_id}' in parents and trashed = false"
            results = self.drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)',
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            
            items = results.get('files', [])
            if not items:
                logger.error("""
Colab notebook bulunamadı! Lütfen aşağıdakileri kontrol edin:
1. 'train_models.ipynb' dosyası Google Drive'ınızda trading_bot klasöründe mevcut mu?
2. config/colab_config.json dosyasındaki ID'ler doğru mu?
3. Google Drive API izinleri doğru ayarlanmış mı?

Çözüm için:
1. Örnek Colab notebook'u Drive'ınızdaki trading_bot klasörüne yükleyin
2. colab_config.json dosyasını güncelleyin
3. Google Cloud Console'dan API izinlerini kontrol edin""")
                raise Exception("Colab eğitimi başlatma hatası!")
            
            notebook_id = items[0]['id']
            self.config['colab_notebook_id'] = notebook_id
            
            # Training status dosyasını oluştur/güncelle
            status_data = {
                'status': 'started',
                'start_time': datetime.now().isoformat(),
                'model_folder_id': trading_bot_folder_id,  # Artık trading_bot klasörünü kullan
                'data_folder_id': trading_bot_folder_id    # Artık trading_bot klasörünü kullan
            }
            
            # Status dosyasını ara (trading_bot klasörü içinde)
            query = f"name = '{self.config['model_files']['status']}' and '{trading_bot_folder_id}' in parents and trashed = false"
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
                # Yeni dosya oluştur (trading_bot klasörü içinde)
                file_metadata = {
                    'name': self.config['model_files']['status'],
                    'parents': [trading_bot_folder_id]  # trading_bot klasörüne kaydet
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
            
            logger.info("[BASARILI] Colab eğitimi başlatıldı!")
            return True
            
        except Exception as e:
            logger.error(f"[HATA] Colab eğitim hatası: {str(e)}")
            return False
            
    def download_model(self, model_name='lstm_model.pth', save_path='saved_models/'):
        """Download trained model from Drive"""
        try:
            trading_bot_folder_id = self.config['drive_folders'].get('trading_bot')
            if not trading_bot_folder_id:
                logger.error("Trading bot klasör ID'si bulunamadı!")
                return False

            # Model dosyasını trading_bot klasöründe ara
            results = self.drive_service.files().list(
                q=f"name='{model_name}' and '{trading_bot_folder_id}' in parents and trashed = false",
                fields="files(id, name, modifiedTime)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
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
            trading_bot_folder_id = self.config['drive_folders'].get('trading_bot')
            if not trading_bot_folder_id:
                logger.error("Trading bot klasör ID'si bulunamadı!")
                return False

            # Status dosyasını trading_bot klasöründe kontrol et
            results = self.drive_service.files().list(
                q=f"name='training_status.json' and '{trading_bot_folder_id}' in parents and trashed = false",
                fields="files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
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
            self.config = self._read_json_file(self.config_path, "Colab Config")
            if not self.config:
                raise Exception("Colab konfigürasyonu yüklenemedi!")

            # Klasörleri kontrol et ve gerekirse oluştur
            self._setup_drive_folders()
            
            # Colab notebook'u kontrol et ve yükle
            notebook_info = self._setup_colab_notebook()
            if notebook_info:
                self.config['colab_notebook_id'] = notebook_info['id']
                # Konfigürasyonu güncelle
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=4)

            self.colab_notebook_id = self.config.get('colab_notebook_id')
            self.model_folder_id = self.config.get('drive_folders', {}).get('models')
            self.data_folder_id = self.config.get('drive_folders', {}).get('data')
            # Colab URL'ini oluştur
            self.colab_url = f"https://colab.research.google.com/drive/{self.colab_notebook_id}"
            logger.info("[BILGI] Colab konfigürasyonu başarıyla yüklendi")
        except Exception as e:
            logger.error(f"[HATA] Colab konfigürasyonu yüklenirken hata: {str(e)}")
            raise

    def _setup_colab_notebook(self):
        """Colab notebook'u kontrol et ve gerekirse yükle"""
        try:
            print("\n==================================================")
            print("COLAB NOTEBOOK KONTROLÜ")
            print("==================================================")
            
            # Önce mevcut notebook'u kontrol et ve sil
            if self.config.get('colab_notebook_id'):
                try:
                    self.drive_service.files().delete(
                        fileId=self.config['colab_notebook_id']
                    ).execute()
                    print("ℹ Eski notebook silindi")
                except Exception:
                    print("ℹ Eski notebook zaten silinmiş")
            
            # trading_bot klasör ID'sini al
            trading_bot_id = self.config.get('drive_folders', {}).get('trading_bot')
            if not trading_bot_id:
                raise Exception("trading_bot klasör ID'si bulunamadı")
            
            print("ℹ Yerel notebook dosyası kontrol ediliyor...")
            
            # Yerel notebook dosyasını kontrol et
            notebook_path = os.path.join(self.base_dir, 'notebooks', 'train_lstm.ipynb')
            if not os.path.exists(notebook_path):
                print("❌ Yerel notebook dosyası bulunamadı!")
                print("\nÖnerilen çözümler:")
                print("1. notebooks/train_lstm.ipynb dosyasının varlığını kontrol edin")
                print("2. Projeyi GitHub'dan yeniden klonlayın")
                raise Exception("Notebook dosyası bulunamadı")
            
            print("✓ Yerel notebook dosyası bulundu")
            print("ℹ Notebook Drive'a yükleniyor...")
            
            # Notebook'u Drive'a yükle
            file_metadata = {
                'name': 'train_lstm.ipynb',
                'parents': [trading_bot_id],
                'mimeType': 'application/vnd.google.colab'
            }
            
            media = MediaFileUpload(
                notebook_path,
                mimetype='application/json',
                resumable=True
            )
            
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name',
                supportsAllDrives=True
            ).execute()
            
            print(f"✅ Notebook başarıyla yüklendi (ID: {file['id']})")
            print(f"ℹ Colab URL: https://colab.research.google.com/drive/{file['id']}")
            
            return file
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Notebook yükleme hatası: {error_msg}")
            print("\nÖnerilen çözümler:")
            print("1. İnternet bağlantınızı kontrol edin")
            print("2. Google Drive API izinlerini kontrol edin")
            print("3. Drive'da yeterli alan olduğundan emin olun")
            logger.error(f"[HATA] Notebook yükleme hatası: {str(e)}")
            return None

    def _setup_drive_folders(self):
        """Drive'da gerekli klasörleri kontrol et ve oluştur"""
        try:
            print("\n==================================================")
            print("GOOGLE DRIVE KLASÖR KONTROLÜ")
            print("==================================================")
            print("Drive'daki klasörler kontrol ediliyor...")
            
            # Ana klasörü kontrol et/oluştur
            trading_bot_folder = self._get_or_create_folder('trading_bot')
            if not trading_bot_folder:
                print("❌ trading_bot klasörü oluşturulamadı!")
                print("\nÖnerilen çözümler:")
                print("1. Google Drive API izinlerinizi kontrol edin")
                print("2. Drive'da yeterli alan olduğundan emin olun")
                print("3. İnternet bağlantınızı kontrol edin")
                raise Exception("trading_bot klasörü oluşturulamadı!")
            
            # Alt klasörleri oluştur
            data_folder = self._get_or_create_folder('data', parent_id=trading_bot_folder['id'])
            if not data_folder:
                print("❌ data klasörü oluşturulamadı!")
                raise Exception("data klasörü oluşturulamadı!")
                
            models_folder = self._get_or_create_folder('models', parent_id=trading_bot_folder['id'])
            if not models_folder:
                print("❌ models klasörü oluşturulamadı!")
                raise Exception("models klasörü oluşturulamadı!")
            
            # Klasör ID'lerini güncelle
            self.config['drive_folders'] = {
                'root': trading_bot_folder['id'],
                'models': models_folder['id'],
                'data': data_folder['id'],
                'trading_bot': trading_bot_folder['id']
            }
            
            # Konfigürasyonu kaydet
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            print("\nKlasör yapısı:")
            print(f"└── trading_bot/ (ID: {trading_bot_folder['id']})")
            print(f"    ├── data/     (ID: {data_folder['id']})")
            print(f"    └── models/   (ID: {models_folder['id']})")
            print("\n✅ Drive klasörleri hazır!")
            
            logger.info("[BILGI] Drive klasörleri başarıyla oluşturuldu/güncellendi")
            return True
            
        except Exception as e:
            logger.error(f"[HATA] Drive klasörleri oluşturulurken hata: {str(e)}")
            return False
            
    def _get_or_create_folder(self, folder_name, parent_id=None):
        """Drive'da klasör kontrol et veya oluştur"""
        try:
            # Önce klasörü ara
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            if parent_id:
                query += f" and '{parent_id}' in parents"
            query += " and trashed=false"
            
            results = self.drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)',
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            
            files = results.get('files', [])
            
            # Klasör varsa
            if files:
                print(f"✓ {folder_name}/ klasörü bulundu (ID: {files[0]['id']})")
                logger.info(f"[BILGI] {folder_name} klasörü bulundu (ID: {files[0]['id']})")
                return files[0]
            
            # Klasör yoksa oluştur
            print(f"ℹ {folder_name}/ klasörü bulunamadı, oluşturuluyor...")
            
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_id:
                folder_metadata['parents'] = [parent_id]
            
            folder = self.drive_service.files().create(
                body=folder_metadata,
                fields='id, name',
                supportsAllDrives=True
            ).execute()
            
            print(f"✅ {folder_name}/ klasörü oluşturuldu (ID: {folder['id']})")
            logger.info(f"[BILGI] {folder_name} klasörü oluşturuldu (ID: {folder['id']})")
            return folder
            
        except Exception as e:
            error_msg = str(e)
            if "insufficientFilePermissions" in error_msg:
                print(f"\n❌ {folder_name}/ klasörü için yetki hatası!")
                print("\nÖnerilen çözümler:")
                print("1. Google Cloud Console'da Drive API'yi etkinleştirin")
                print("2. OAuth onay ekranını yapılandırın")
                print("3. credentials.json dosyasını yeniden indirin")
            elif "dailyLimitExceeded" in error_msg:
                print(f"\n❌ {folder_name}/ klasörü için API limit hatası!")
                print("\nÖnerilen çözümler:")
                print("1. Birkaç dakika bekleyip tekrar deneyin")
                print("2. Google Cloud Console'dan API limitlerini kontrol edin")
            else:
                print(f"\n❌ {folder_name}/ klasörü oluşturulurken hata: {error_msg}")
            
            logger.error(f"[HATA] {folder_name} klasörü oluşturulurken hata: {str(e)}")
            return None 