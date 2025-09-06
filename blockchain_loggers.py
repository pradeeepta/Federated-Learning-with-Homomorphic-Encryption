import hashlib
import json
import os
from web3 import Web3
from datetime import datetime
from dotenv import load_dotenv

# Load private key from .env
load_dotenv()
private_key = os.getenv("PRIVATE_KEY")
if not private_key:
    raise Exception("❌ PRIVATE_KEY not found in .env file!")

# Connect to Ganache
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
account = web3.eth.account.from_key(private_key).address


# Load contract ABI and address
with open("contract_abi.json") as f:
    contract_abi = json.load(f)

with open("contract_address.txt", "r") as f:
    contract_address = f.read().strip()

contract = web3.eth.contract(address=contract_address, abi=contract_abi)

def hash_model_weights(weights_bytes):
    hash_obj = hashlib.sha256()
    for layer in weights_bytes:
        hash_obj.update(layer)
    return hash_obj.hexdigest()


def log_model_update(hospital_id, weights_bytes, accuracy, epoch, dataset_slice="slice_1"):
    weight_hash = hash_model_weights(weights_bytes)
    timestamp = datetime.utcnow().isoformat()

    print("📝 Logging to blockchain...")
    tx = contract.functions.logUpdate(
        hospital_id,
        weight_hash,
        timestamp,
        str(epoch),
        dataset_slice,
        str(round(accuracy, 2))
    ).build_transaction({
        'from': account,
        'nonce': web3.eth.get_transaction_count(account),
        'gas': 3000000,
        'gasPrice': web3.to_wei('20', 'gwei')
    })

    # Sign and send the transaction
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)

    print(f"✅ Blockchain log submitted. Tx Hash: {web3.to_hex(tx_hash)}")


# Test
if __name__ == "__main__":
    dummy_weights = [b"layer1", b"layer2"]
    log_model_update("hospitalXYZ", dummy_weights, 95.47, epoch="6")
