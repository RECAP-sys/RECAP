DEFAULT_GRPC_MESSAGE_SIZE_MB = 512
DEFAULT_GRPC_MESSAGE_SIZE_BYTES = DEFAULT_GRPC_MESSAGE_SIZE_MB * 1024 * 1024


def grpc_message_options(max_message_length: int = DEFAULT_GRPC_MESSAGE_SIZE_BYTES):
    limit = int(max_message_length)
    return [
        ("grpc.max_send_message_length", limit),
        ("grpc.max_receive_message_length", limit),
        ("grpc.keepalive_time_ms", 60_000),
        ("grpc.keepalive_timeout_ms", 120_000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_time_between_pings_ms", 30_000),
        ("grpc.http2.min_ping_interval_without_data_ms", 30_000),
    ]
