from web3 import Web3
from datetime import datetime
import hashlib
import json

# ========= 🌐 Connect to Ganache =========
ganache_url = "http://127.0.0.1:7545"
w3 = Web3(Web3.HTTPProvider(ganache_url))

# ========= 🔐 Use Your Dev Account =========
account_address = "0x0a7eefcd2AfCd4496a4F2eCBE89e8403F1Ed01c3"
private_key = "0xd97d1dc7ac4ab75569cce96f3f28a8a5e8d01658d8e7cae464c2de7639360433"

# ========= 🔗 Load Contract =========
contract_address = "0x0a7eefcd2AfCd4496a4F2eCBE89e8403F1Ed01c3"  
with open("ModelLoggerABI.json", "r") as f:
    contract_abi = json.load(f)

contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# ========= 📝 Data to Log =========
client_id = "Apollo_Hospital"
model_bytes = b"model_weights_dummy" 
hash_digest = hashlib.sha256(model_bytes).hexdigest()
timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
epoch_info = "epoch=5, slice=10%"

# ========= 🚀 Send Transaction =========
nonce = w3.eth.get_transaction_count(account_address)

txn = contract.functions.storeLog(client_id, hash_digest, timestamp, epoch_info).build_transaction({
    'from': account_address,
    'nonce': nonce,
    'gas': 500000,
    'gasPrice': w3.to_wei('1', 'gwei')
})

signed_txn = w3.eth.account.sign_transaction(txn, private_key=private_key)
tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print("✅ Logged successfully on blockchain.")
print("🔗 Tx Hash:", receipt.transactionHash.hex())
