from web3 import Web3
import json

# Connect to Ganache
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
print("🔌 Connected:", web3.is_connected())

# Load contract ABI
with open("contract_abi.json", "r") as f:
    abi = json.load(f)

# Paste the contract address you think was deployed
contract_address = "0x66e9C792FDCCfd486238cD49aea557A285e30a3B"

# Check if contract exists at that address
code = web3.eth.get_code(contract_address)

if code == b'' or code == '0x':
    print("❌ No contract found at this address!")
else:
    print("✅ Contract is deployed at this address.")
