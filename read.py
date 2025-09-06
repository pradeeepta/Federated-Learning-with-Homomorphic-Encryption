from web3 import Web3

# Connect to Ganache
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Replace with your contract address and ABI
contract_address = web3.to_checksum_address("0xcE2964302ef50Dc64d5cb2922751F43DF0E0fD8e")

contract_abi = [  # Only relevant read functions
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
]

# Instantiate the contract
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Get total number of logs
total_updates = contract.functions.getTotalUpdates().call()
print(f"📦 Total Blockchain Logs: {total_updates}\n")

# Iterate and print each update
for i in range(total_updates):
    update = contract.functions.getUpdate(i).call()
    print(f"🧾 Log #{i+1}")
    print(f"   🏥 Hospital ID   : {update[0]}")
    print(f"   🔐 Model Hash    : {update[1]}")
    print(f"   🕒 Timestamp     : {update[2]}")
    print(f"   🌀 Epoch         : {update[3]}")
    print(f"   🧬 Dataset Slice : {update[4]}")
    print(f"   🎯 Accuracy      : {update[5]}")
    print("------------------------------------------------")
