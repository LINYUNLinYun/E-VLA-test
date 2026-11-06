"""Framed trusted-local RPC, also used for the in-request observation exchange."""
import pickle
import struct

MAX_BYTES = 128 * 1024 * 1024


def _exact(connection, size):
    result = bytearray()
    while len(result) < size:
        block = connection.recv(size - len(result))
        if not block:
            raise ConnectionError("Online RPC ended before its frame completed")
        result.extend(block)
    return bytes(result)


def send(connection, value):
    payload = pickle.dumps(value, protocol=5)
    if len(payload) > MAX_BYTES:
        raise ValueError("Oversized online RPC frame")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def receive(connection):
    size = struct.unpack("!I", _exact(connection, 4))[0]
    if size > MAX_BYTES:
        raise ValueError("Oversized online RPC frame")
    return pickle.loads(_exact(connection, size))
