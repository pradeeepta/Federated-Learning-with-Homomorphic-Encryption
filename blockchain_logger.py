import os
import json
import base64
import hashlib
from datetime import datetime
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

# Use a 16-byte AES key (in production, use .env or key vault)
AES_KEY = b'1234567890123456'  
BLOCKCHAIN_LOG_FILE = 'blockchain_audit_log.json'

def encrypt_hash(text):
    cipher = AES.new(AES_KEY, AES.MODE_ECB)
    encrypted = cipher.encrypt(pad(text.encode(), AES.block_size))
    return base64.b64encode(encrypted).decode()

def log_to_blockchain(hospital_id, prediction, confidence, test_type, test_time):
    log_entry = {
        "hospital_id": hospital_id,
        "prediction": prediction,
        "confidence": confidence,
        "test_type": test_type,
        "timestamp": test_time.isoformat()
    }

    # Generate raw hash and encrypt it
    log_string = json.dumps(log_entry, sort_keys=True)
    raw_hash = hashlib.sha256(log_string.encode()).hexdigest()
    encrypted_hash = encrypt_hash(raw_hash)
    log_entry['hash'] = encrypted_hash

    # Append to blockchain log file
    if os.path.exists(BLOCKCHAIN_LOG_FILE):
        with open(BLOCKCHAIN_LOG_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append(log_entry)

    with open(BLOCKCHAIN_LOG_FILE, 'w') as f:
        json.dump(data, f, indent=4)
