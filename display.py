from web3 import Web3
import json
from datetime import datetime

# Connect to Ganache
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Load ABI & Address
with open("model_logger_abi.json", "r") as f:
    abi = json.load(f)

with open("contract_address.txt", "r") as f:
    address = f.read().strip()

contract = web3.eth.contract(address=address, abi=abi)

def fetch_logs():
    print("\n📄 Blockchain Model Update Logs:\n")
    
    logs = contract.events.ModelUpdateLogged().get_logs(from_block=0)

    for log in logs:
        args = log['args']
        print(f"🏥 Hospital ID: {args['hospitalID']}")
        print(f"🔐 Model Hash : {args['modelHash']}")
        print(f"🎯 Accuracy    : {args['accuracy']}")
        print(f"📆 Epoch       : {args['epoch']}")
        print(f"🕒 Timestamp   : {datetime.fromtimestamp(args['timestamp'])}")
        print("-" * 40)

# Run it
fetch_logs()
