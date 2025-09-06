from blockchain_loggers import log_model_update

dummy_weights = [b"layer1weights", b"layer2weights"]

log_model_update(
    hospital_id="hospitalA",
    weights_bytes=dummy_weights,
    accuracy=87.25,
    epoch="3"
)
