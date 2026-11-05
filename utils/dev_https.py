"""
Certificato self-signed per servire ReelForge in HTTPS su localhost.

Il portale TikTok accetta solo Redirect URI che iniziano con https://:
generiamo un certificato locale per "localhost" (valido 5 anni) e lo
salviamo in data/ssl/. Il browser mostrera' un avviso "non sicuro" alla
prima visita (normale per certificati self-signed) — basta premere
Avanzate > Procedi.
"""
import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

DATA_DIR = Path('data')
SSL_DIR = DATA_DIR / 'ssl'
CERT_FILE = SSL_DIR / 'localhost.crt'
KEY_FILE = SSL_DIR / 'localhost.key'


def ensure_local_cert():
    """Genera (una volta) certificato e chiave per localhost. Torna (cert, key) o None."""
    try:
        if CERT_FILE.exists() and KEY_FILE.exists():
            return str(CERT_FILE), str(KEY_FILE)

        SSL_DIR.mkdir(parents=True, exist_ok=True)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, 'localhost'),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, 'ReelForge local dev'),
        ])
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)  # self-signed
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365 * 5))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName('localhost'),
                    x509.DNSName('localhost.localdomain'),
                    x509.IPAddress(ipaddress.ip_address('127.0.0.1')),
                    x509.IPAddress(ipaddress.ip_address('::1')),
                ]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        KEY_FILE.write_bytes(
            key.private_bytes(serialization.Encoding.PEM,
                              serialization.PrivateFormat.TraditionalOpenSSL,
                              serialization.NoEncryption())
        )
        print('[HTTPS] Certificato self-signed generato in data/ssl/')
        return str(CERT_FILE), str(KEY_FILE)
    except Exception as e:
        print(f'[HTTPS] Certificato non generato ({e}): si continua in HTTP')
        return None


if __name__ == '__main__':
    res = ensure_local_cert()
    print('ssl_context:', res)
