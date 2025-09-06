// AuditLogger.sol
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
