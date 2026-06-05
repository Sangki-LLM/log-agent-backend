import base64
import os

from cryptography.fernet import Fernet

from app.core.config import settings


def _get_fernet() -> Fernet:
    key = settings.pem_encryption_key
    if not key:
        # 개발 환경 전용 임시 키 (운영 시 반드시 .env에 설정)
        key = base64.urlsafe_b64encode(b"dev-only-key-do-not-use-in-prod!")
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_pem(pem_text: str) -> str:
    return _get_fernet().encrypt(pem_text.encode()).decode()


def decrypt_pem(encrypted: str) -> str:
    return _get_fernet().decrypt(encrypted.encode()).decode()
