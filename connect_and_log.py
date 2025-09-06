import json
import hashlib
import os
from web3 import Web3
from datetime import datetime
from dotenv import load_dotenv

# ===== Load Environment Variables =====
load_dotenv()

# ===== Connect to Ganache =====
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
if not web3.is_connected():
    raise Exception("❌ Unable to connect to Ganache")
print("✅ Connected to Ganache")

# ===== Load ABI from separate JSON file =====
with open("contract_abi.json", "r") as abi_file:
    contract_abi = json.load(abi_file)

# ===== Load contract address from file =====
with open("contract_address.txt", "r") as addr_file:
    contract_address = web3.to_checksum_address(addr_file.read().strip())

# ===== Connect to deployed contract =====
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# ===== Load Private Keys from .env =====
PRIVATE_KEYS = {
    "hospital1": os.getenv("PRIVATE_KEY_HOSPITAL_1"),
    "hospital2": os.getenv("PRIVATE_KEY_HOSPITAL_2"),
    "hospital3": os.getenv("PRIVATE_KEY_HOSPITAL_3"),
    "manipal": os.getenv("PRIVATE_KEY_HOSPITAL_1"),  # reuse hospital1
}

# ===== Ganache accounts =====
ACCOUNTS = {
    "hospital1": web3.eth.accounts[0],
    "hospital2": web3.eth.accounts[1],
    "hospital3": web3.eth.accounts[2],
    "manipal": web3.eth.accounts[0],
}

# ===== Function to hash model weights =====
def hash_weights(weights_bytes):
    combined = b''.join(weights_bytes)
    return hashlib.sha256(combined).hexdigest()

# ===== Blockchain Logger Function =====
def log_model_update(hospital_id, weights_bytes, accuracy, epoch, dataset_slice="lung"):
    try:
        model_hash = hash_weights(weights_bytes)
        timestamp = str(datetime.now())
        accuracy_str = f"{accuracy:.4f}"
        hospital_key = hospital_id.lower()

        print(f"\n📦 Model Hash: {model_hash}")

        account = ACCOUNTS.get(hospital_key)
        private_key = PRIVATE_KEYS.get(hospital_key)

        if not account or not private_key:
            print(f"❌ Missing account/private key for hospital: {hospital_id}")
            return

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

        print(f"📝 Sent Tx: {web3.to_hex(tx_hash)} — waiting for confirmation...")
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        print("✅ Transaction mined successfully!")

    except Exception as e:
        print(f"❌ Logging failed: {e}")

# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    dummy_weights = [b"layer1_weights", b"layer2_weights", b"layer3_bias"]
    log_model_update("hospital1", dummy_weights, accuracy=0.9273, epoch=5)
