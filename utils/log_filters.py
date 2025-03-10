import logging
import re

class SensitiveDataFilter(logging.Filter):
    """Hassas bilgileri log kayıtlarından filtreleyen sınıf"""
    
    def __init__(self):
        super().__init__()
        # Hassas bilgi kalıpları
        self.patterns = [
            (r'password[\'":\s]+[^\s,}]+', 'password: ***'),
            (r'login[\'":\s]+\d+', lambda m: re.sub(r'\d+', '***', m.group())),
            (r'api[_-]?key[\'":\s]+[^\s,}]+', 'api_key: ***'),
            (r'secret[\'":\s]+[^\s,}]+', 'secret: ***'),
            (r'token[\'":\s]+[^\s,}]+', 'token: ***'),
            (r'\b\d{16}\b', '****-****-****-****'),  # Kredi kartı numaraları
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email@***.com'),  # Email adresleri
        ]

    def filter(self, record):
        if isinstance(record.msg, str):
            msg = record.msg
            for pattern, replacement in self.patterns:
                if callable(replacement):
                    msg = re.sub(pattern, replacement, msg)
                else:
                    msg = re.sub(pattern, replacement, msg)
            record.msg = msg
        return True 