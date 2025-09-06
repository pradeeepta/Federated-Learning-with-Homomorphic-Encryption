from web3 import Web3
import json

# Connect to Ganache or local network
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))

# Load ABI
with open("ModelLoggerABI.json", "r") as f:
    abi = json.load(f)

# Your deployed contract address
address = "0x4E85446Ed98de669FA9aF878448798Bd8FFE6987"
contract = w3.eth.contract(address=address, abi=abi)

# Example: Call getTotalUpdates
print("Total updates:", contract.functions.getTotalUpdates().call())
