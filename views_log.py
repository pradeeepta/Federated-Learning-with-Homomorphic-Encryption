import json
from web3 import Web3
from dotenv import load_dotenv
import os

# ✅ Load .env if needed (optional for future extension)
load_dotenv()

# 🔗 Connect to Ganache
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
if not web3.is_connected():
    raise Exception("❌ Unable to connect to Ganache")

# 🔐 Load ABI and contract address
with open("contract_abi.json", "r") as abi_file:
    contract_abi = json.load(abi_file)

with open("contract_address.txt", "r") as addr_file:
    contract_address = web3.to_checksum_address(addr_file.read().strip())

# 🧱 Load contract instance
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# 🔍 Get total updates
total_updates = contract.functions.getTotalUpdates().call()
print(f"\n🧾 Total Updates Logged on Blockchain: {total_updates}\n")

# 📋 Fetch and display all updates
for index in range(total_updates):
    update = contract.functions.getUpdate(index).call()
    
    print(f"🔹 Update #{index+1}")
    print(f"  🏥 Hospital ID     : {update[0]}")
    print(f"  🧬 Model Hash      : {update[1]}")
    print(f"  🕒 Timestamp       : {update[2]}")
    print(f"  📈 Epoch           : {update[3]}")
    print(f"  🧪 Dataset Slice   : {update[4]}")
    print(f"  🎯 Accuracy        : {update[5]}%\n")
