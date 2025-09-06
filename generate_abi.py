from solcx import compile_standard, install_solc
import json

# Install specific version of solc
install_solc("0.8.0")

# Read the contract
with open("AuditLogger.sol", "r") as file:
    contract_source = file.read()

compiled_sol = compile_standard({
    "language": "Solidity",
    "sources": {
        "AuditLogger.sol": {
            "content": contract_source
        }
    },
    "settings": {
        "outputSelection": {
            "*": {
                "*": ["abi", "metadata", "evm.bytecode"]
            }
        }
    }
}, solc_version="0.8.0")

# Extract and save ABI
abi = compiled_sol["contracts"]["AuditLogger.sol"]["AuditLogger"]["abi"]
with open("contract_abi.json", "w") as f:
    json.dump(abi, f, indent=4)

print("✅ ABI generated and saved to contract_abi.json")
