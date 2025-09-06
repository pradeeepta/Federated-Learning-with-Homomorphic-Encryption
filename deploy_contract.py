# from solcx import compile_source, install_solc
# from web3 import Web3
# import json
# from dotenv import load_dotenv
# import os

# # 📌 Load .env variables
# load_dotenv()

# # ✅ Load all 3 private keys
# private_keys = {
#     "HOSPITAL_1": os.getenv("PRIVATE_KEY_HOSPITAL_1"),
#     "HOSPITAL_2": os.getenv("PRIVATE_KEY_HOSPITAL_2"),
#     "HOSPITAL_3": os.getenv("PRIVATE_KEY_HOSPITAL_3"),
# }

# # ❌ Check if any keys are missing
# for hospital, key in private_keys.items():
#     if not key:
#         raise Exception(f"❌ Private key for {hospital} not found in .env file!")

# # 🟢 Use one specific key (e.g., HOSPITAL_1)
# active_hospital = "HOSPITAL_1"
# private_key = private_keys[active_hospital]
# print(f"🔐 Using private key for: {active_hospital}")

# # 🔌 Connect to Ganache
# web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
# assert web3.is_connected(), "❌ Ganache is not running!"
# print("🔌 Connected to Ganache")

# # 👤 Use the first account from Ganache
# account = web3.eth.accounts[0]
# print(f"👤 Using account: {account}")

# # 🧾 Solidity smart contract code
# contract_source = '''
# // SPDX-License-Identifier: UNLICENSED
# pragma solidity ^0.8.0;

# contract AuditLogger {
#     struct UpdateLog {
#         string hospitalId;
#         string modelHash;
#         string timestamp;
#         string epoch;
#         string datasetSlice;
#         string accuracy;
#     }

#     UpdateLog[] public updates;

#     function logUpdate(
#         string memory hospitalId,
#         string memory modelHash,
#         string memory timestamp,
#         string memory epoch,
#         string memory datasetSlice,
#         string memory accuracy
#     ) public {
#         updates.push(UpdateLog(hospitalId, modelHash, timestamp, epoch, datasetSlice, accuracy));
#     }

#     function getUpdate(uint index) public view returns (UpdateLog memory) {
#         return updates[index];
#     }

#     function getTotalUpdates() public view returns (uint) {
#         return updates.length;
#     }
# }
# '''

# # 🛠️ Install and use Solidity 0.8.0
# install_solc('0.8.0')

# # 🔍 Compile the contract
# compiled_sol = compile_source(contract_source, solc_version='0.8.0')
# contract_id, contract_interface = compiled_sol.popitem()

# # 🧱 Get bytecode and ABI
# bytecode = contract_interface['bin']
# abi = contract_interface['abi']

# # 💾 Save ABI to file
# with open("contract_abi.json", "w") as f:
#     json.dump(abi, f)
#     print("📄 ABI saved to contract_abi.json")

# # 🔨 Create contract constructor
# AuditLogger = web3.eth.contract(abi=abi, bytecode=bytecode)

# # 🧾 Build deployment transaction
# nonce = web3.eth.get_transaction_count(account)
# tx = AuditLogger.constructor().build_transaction({
#     'from': account,
#     'nonce': nonce,
#     'gas': 5000000,
#     'gasPrice': web3.to_wei('20', 'gwei')
# })

# # 🔐 Sign and send transaction
# signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
# tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)

# print(f"⏳ Waiting for transaction {web3.to_hex(tx_hash)} to be mined...")
# tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

# # 📌 Get deployed contract address
# contract_address = tx_receipt.contractAddress
# print(f"✅ Contract deployed at: {contract_address}")

# # 💾 Save contract address to file
# with open("contract_address.txt", "w") as f:
#     f.write(contract_address)
#     print("📄 Contract address saved to contract_address.txt")

from solcx import compile_source, install_solc
from web3 import Web3
import json
from dotenv import load_dotenv
import os

# 📌 Load .env variables
load_dotenv()

# ✅ Load all 3 private keys
private_keys = {
    "HOSPITAL_1": os.getenv("PRIVATE_KEY_HOSPITAL_1"),
    "HOSPITAL_2": os.getenv("PRIVATE_KEY_HOSPITAL_2"),
    "HOSPITAL_3": os.getenv("PRIVATE_KEY_HOSPITAL_3"),
}

# ❌ Check if any keys are missing
for hospital, key in private_keys.items():
    if not key:
        raise Exception(f"❌ Private key for {hospital} not found in .env file!")

# 🔌 Connect to Ganache
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
assert web3.is_connected(), "❌ Ganache is not running!"
print("🔌 Connected to Ganache")

# 🧾 Solidity contract source
contract_source = '''
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract AuditLogger {
    struct UpdateLog {
        string hospitalId;
        string modelHash;
        string timestamp;
        string epoch;
        string datasetSlice;
        string accuracy;
    }

    UpdateLog[] public updates;

    function logUpdate(
        string memory hospitalId,
        string memory modelHash,
        string memory timestamp,
        string memory epoch,
        string memory datasetSlice,
        string memory accuracy
    ) public {
        updates.push(UpdateLog(hospitalId, modelHash, timestamp, epoch, datasetSlice, accuracy));
    }

    function getUpdate(uint index) public view returns (UpdateLog memory) {
        return updates[index];
    }

    function getTotalUpdates() public view returns (uint) {
        return updates.length;
    }
}
'''

# 🛠️ Install and use Solidity 0.8.0
install_solc('0.8.0')

# 🔍 Compile the contract
compiled_sol = compile_source(contract_source, solc_version='0.8.0')
contract_id, contract_interface = compiled_sol.popitem()

# 🧱 Get bytecode and ABI
bytecode = contract_interface['bin']
abi = contract_interface['abi']

# 💾 Save ABI to file (once)
with open("contract_abi.json", "w") as f:
    json.dump(abi, f)
    print("📄 ABI saved to contract_abi.json")

# 🔁 Deploy the contract from each hospital
for hospital_name, private_key in private_keys.items():
    print(f"\n🚀 Deploying contract for: {hospital_name}")
    
    # 🔐 Get account address from private key
    account = web3.eth.account.from_key(private_key).address
    print(f"👤 Derived account: {account}")
    
    # 🔨 Create contract instance
    AuditLogger = web3.eth.contract(abi=abi, bytecode=bytecode)

    # 🧾 Build deployment transaction
    nonce = web3.eth.get_transaction_count(account)
    tx = AuditLogger.constructor().build_transaction({
        'from': account,
        'nonce': nonce,
        'gas': 5000000,
        'gasPrice': web3.to_wei('20', 'gwei')
    })

    # 🔐 Sign and send transaction
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)

    print(f"⏳ Waiting for transaction {web3.to_hex(tx_hash)} to be mined...")
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

    # ✅ Get deployed contract address
    contract_address = tx_receipt.contractAddress
    print(f"✅ {hospital_name} contract deployed at: {contract_address}")

    # 💾 Save address to separate file
    with open(f"{hospital_name.lower()}_contract_address.txt", "w") as f:
        f.write(contract_address)
        print(f"📄 Contract address saved to {hospital_name.lower()}_contract_address.txt")
