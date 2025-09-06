from web3 import Web3
import hashlib
import json
from datetime import datetime

ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
account = web3.eth.accounts[0]

# Load ABI & Contract
with open("model_logger_abi.json", "r") as f:
    abi = json.load(f)
with open("contract_address.txt", "r") as f:
    address = f.read().strip()

contract = web3.eth.contract(address=address, abi=abi)

def log_model_update(hospital_id, weights_bytes, accuracy, epoch):
    # 1. Hash the weights
    full_weights = b''.join(weights_bytes)
    model_hash = hashlib.sha256(full_weights).hexdigest()

    # 2. Send TX
    tx = contract.functions.logUpdate(
        hospital_id,
        model_hash,
        str(round(accuracy, 4)),
        str(epoch)
    ).transact({'from': account})

    receipt = web3.eth.wait_for_transaction_receipt(tx)
    print("✅ Logged on blockchain. TxHash:", web3.to_hex(receipt.transactionHash))
    
if __name__ == "__main__":
    # Dummy weights
    dummy_weights = [b"layer1weights", b"layer2weights"]

    # Call the logger
    log_model_update(
        hospital_id="hospitalA",
        weights_bytes=dummy_weights,
        accuracy=87.25,
        epoch="3"
    )
