import logging

class SensitiveDataFilter(logging.Filter):
    """
    Hassas verileri loglarda maskelemek için filtre
    """
    def __init__(self):
        super().__init__()
        self.sensitive_patterns = [
            'password',
            'api_key',
            'secret',
            'token',
            'login',
            'credential'
        ]
    
    def filter(self, record):
        """
        Log mesajlarındaki hassas verileri maskele
        
        Parametreler:
        - record: Log kaydı
        
        Dönüş:
        - bool: Her zaman True (log kaydı her zaman geçer, sadece içeriği değiştirilir)
        """
        try:
            message = record.getMessage()
            
            # Hassas verileri maskele
            for pattern in self.sensitive_patterns:
                if pattern in message.lower():
                    # Değeri bul ve maskele
                    parts = message.split(pattern + "=")
                    if len(parts) > 1:
                        # İkinci parçadaki değeri maskele
                        value_end = parts[1].find(" ")
                        if value_end == -1:
                            value_end = len(parts[1])
                        
                        value = parts[1][:value_end]
                        masked_value = '*' * len(value)
                        record.msg = record.msg.replace(f"{pattern}={value}", f"{pattern}={masked_value}")
                    
                    # Diğer olası formatları da kontrol et
                    parts = message.split(f'"{pattern}": "')
                    if len(parts) > 1:
                        value_end = parts[1].find('"')
                        if value_end != -1:
                            value = parts[1][:value_end]
                            masked_value = '*' * len(value)
                            record.msg = record.msg.replace(f'"{pattern}": "{value}"', f'"{pattern}": "{masked_value}"')
            
            return True
            
        except Exception as e:
            # Filtre hatası durumunda orijinal mesajı değiştirmeden geçir
            return True 