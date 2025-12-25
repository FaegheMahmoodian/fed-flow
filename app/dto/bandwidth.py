
class BandWidth:
    """
    Represents network bandwidth.

    Supports two modes:
    1. Measured: bandwidth calculated from actual data transfer
    2. Hardcoded: bandwidth set from predefined values
    """

    def __init__(self, transferred_bytes: float = None, time: float = None,
                 hardcoded_value: float = None, is_hardcoded: bool = False):
        """
        Initialize BandWidth object.

        Args:
            transferred_bytes: Amount of data transferred (bytes)
            time: Time taken for transfer (seconds)
            hardcoded_value: Predefined bandwidth value (bytes/sec)
            is_hardcoded: Whether this is a hardcoded bandwidth
        """
        if hardcoded_value is not None:
            self.bandwidth = hardcoded_value
            self.is_hardcoded = True
        elif transferred_bytes is not None and time is not None and time > 0:
            self.bandwidth = transferred_bytes / time
            self.is_hardcoded = is_hardcoded
        else:
            raise ValueError("Must provide either hardcoded_value or (transferred_bytes, time)")

    def __eq__(self, other):
        return self.bandwidth == other.bandwidth

    def __hash__(self):
        return hash(self.bandwidth)

    def __repr__(self):
        mode = "hardcoded" if self.is_hardcoded else "measured"
        return f"BandWidth({self.to_mbps():.2f} Mbps, {mode})"

    def to_mbps(self) -> float:
        """Convert bandwidth to Mbps (Megabits per second)"""
        return (self.bandwidth * 8) / 1_000_000

    def to_mbytes_per_sec(self) -> float:
        """Convert bandwidth to MB/s (Megabytes per second)"""
        return self.bandwidth / 1_000_000
