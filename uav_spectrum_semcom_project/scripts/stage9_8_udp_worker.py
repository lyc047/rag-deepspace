#!/usr/bin/env python
"""Stage-9.7 UDP worker reused with the Stage-9.8 compact codec."""

from __future__ import annotations

import stage9_7_udp_worker as base

from spectrum_semcom.stage9_8_compact_transport import (
    corrupt_compact_protocol_payload,
    decode_compact_udp_datagram,
    encode_compact_udp_datagram,
)


base.encode_udp_datagram = encode_compact_udp_datagram
base.decode_udp_datagram = decode_compact_udp_datagram
base.corrupt_protocol_payload = corrupt_compact_protocol_payload


if __name__ == "__main__":
    base.main()
