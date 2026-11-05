"""furniture_client: the program-side library of the furniture cell (the proxy, frame decoding, the transports).

    from furniture_client import Cell, CellError, decode_frame, connect_practice, given_dir

The graded run hands solve(cell) a connected Cell; the practice lab is reached with connect_practice().
"""

from .client import (BOOTSTRAP, PRACTICE_SOCKET, PUBLIC_CAMERAS, Cell, CellError, connect_practice, connect_stdio,
                     decode_frame, given_dir, main)

__all__ = ["Cell", "CellError", "decode_frame", "connect_stdio", "connect_practice", "given_dir", "main", "BOOTSTRAP",
           "PUBLIC_CAMERAS", "PRACTICE_SOCKET"]
