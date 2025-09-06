// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ModelLogger {
    struct Update {
        string hospitalID;
        string modelHash;
        string accuracy;
        string epoch;
        uint timestamp;
    }

    Update[] public updates;

    // ✅ Event declaration
    event ModelUpdateLogged(string hospitalID, string modelHash, string accuracy, string epoch, uint timestamp);

    function logUpdate(string memory hospitalID, string memory modelHash, string memory accuracy, string memory epoch) public {
        updates.push(Update(hospitalID, modelHash, accuracy, epoch, block.timestamp));
        
        // ✅ Emit event for blockchain logging
        emit ModelUpdateLogged(hospitalID, modelHash, accuracy, epoch, block.timestamp);
    }

    function getUpdate(uint index) public view returns (string memory, string memory, string memory, string memory, uint) {
        Update memory u = updates[index];
        return (u.hospitalID, u.modelHash, u.accuracy, u.epoch, u.timestamp);
    }

    function getTotalUpdates() public view returns (uint) {
        return updates.length;
    }
}
