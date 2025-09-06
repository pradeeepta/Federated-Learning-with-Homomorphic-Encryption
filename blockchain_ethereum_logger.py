import json
import hashlib
import os
from web3 import Web3
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Connect to Ganache
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
if not web3.is_connected():
    raise Exception("❌ Unable to connect to Ganache")

# Contract ABI
contract_abi = [
    {
        "inputs": [],
        "name": "getTotalUpdates",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "index", "type": "uint256"}],
        "name": "getUpdate",
        "outputs": [
            {
                "components": [
                    {"internalType": "string", "name": "hospitalId", "type": "string"},
                    {"internalType": "string", "name": "modelHash", "type": "string"},
                    {"internalType": "string", "name": "timestamp", "type": "string"},
                    {"internalType": "string", "name": "epoch", "type": "string"},
                    {"internalType": "string", "name": "datasetSlice", "type": "string"},
                    {"internalType": "string", "name": "accuracy", "type": "string"},
                ],
                "internalType": "struct AuditLogger.UpdateLog",
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "string", "name": "hospitalId", "type": "string"},
            {"internalType": "string", "name": "modelHash", "type": "string"},
            {"internalType": "string", "name": "timestamp", "type": "string"},
            {"internalType": "string", "name": "epoch", "type": "string"},
            {"internalType": "string", "name": "datasetSlice", "type": "string"},
            {"internalType": "string", "name": "accuracy", "type": "string"},
        ],
        "name": "logUpdate",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "name": "updates",
        "outputs": [
            {"internalType": "string", "name": "hospitalId", "type": "string"},
            {"internalType": "string", "name": "modelHash", "type": "string"},
            {"internalType": "string", "name": "timestamp", "type": "string"},
            {"internalType": "string", "name": "epoch", "type": "string"},
            {"internalType": "string", "name": "datasetSlice", "type": "string"},
            {"internalType": "string", "name": "accuracy", "type": "string"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

# Contract address (from Ganache deployment)
contract_address = web3.to_checksum_address("0xcE2964302ef50Dc64d5cb2922751F43DF0E0fD8e")
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Private keys from .env
PRIVATE_KEYS = {
    "hospital1": os.getenv("PRIVATE_KEY_HOSPITAL_1"),
    "hospital2": os.getenv("PRIVATE_KEY_HOSPITAL_2"),
    "hospital3": os.getenv("PRIVATE_KEY_HOSPITAL_3"),
    "manipal": os.getenv("PRIVATE_KEY_HOSPITAL_1"),  # Using same key as hospital1 for manipal
}

# Corresponding accounts (Ganache accounts)
from eth_account import Account

ACCOUNTS = {
    "hospital1": Account.from_key(PRIVATE_KEYS["hospital1"]).address,
    "hospital2": Account.from_key(PRIVATE_KEYS["hospital2"]).address,
    "hospital3": Account.from_key(PRIVATE_KEYS["hospital3"]).address,
    "manipal": Account.from_key(PRIVATE_KEYS["hospital1"]).address,
}

# Utility: hash model weights
def hash_weights(weights_bytes):
    combined = b''.join(weights_bytes)
    return hashlib.sha256(combined).hexdigest()

# Blockchain logging function
# blockchain_ethereum_logger.py

def log_model_update(hospital_id, weights_bytes, accuracy, epoch, dataset_slice="lung", model_hash=None):
    try:
        model_hash = hash_weights(weights_bytes)
        timestamp = str(datetime.now())
        accuracy_str = f"{accuracy:.4f}"
        hospital_key = hospital_id.lower()

        account = ACCOUNTS.get(hospital_key)
        private_key = PRIVATE_KEYS.get(hospital_key)

        if not account or not private_key:
            return  # Skip silently

        nonce = web3.eth.get_transaction_count(account, "pending")

        tx = contract.functions.logUpdate(
            hospital_id,
            model_hash,
            timestamp,
            str(epoch),
            dataset_slice,
            accuracy_str
        ).build_transaction({
            "from": account,
            "nonce": nonce,
            "gas": 300000,
            "gasPrice": web3.to_wei("20", "gwei")
        })

        signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)

    except Exception:
        pass  

if __name__ == "__main__":
    from tensorflow.keras.models import load_model
    import numpy as np

    # Load model once
    model = load_model("best_model.h5")
    weights = model.get_weights()
    weights_bytes = [w.astype(np.float32).tobytes() for w in weights]

    # Define hospitals and dummy accuracies
    hospitals = {
        "hospital1": 87.54,
        "hospital2": 83.21,
        "hospital3": 89.76,
    }

    # Log for each hospital
    for hospital, acc in hospitals.items():
        print(f"\n🚀 Logging update for {hospital}")
        log_model_update(
            hospital_id=hospital,
            weights_bytes=weights_bytes,
            accuracy=acc,
            epoch=30,
            dataset_slice="lung"
        )
