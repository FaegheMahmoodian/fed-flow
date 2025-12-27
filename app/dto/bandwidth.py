"""
BandWidth - Network bandwidth representation with dual mode support

Supports:
    - Measured mode: Calculate bandwidth from actual data transfer
    - Hardcoded mode: Use predefined bandwidth values
    - Unit conversions: Mbps, MB/s, bytes/s
"""


class BandWidth:
    """Network bandwidth with measured/hardcoded dual mode support"""

    def __init__(
            self,
            transferred_bytes: float = None,
            time: float = None,
            hardcoded_value: float = None,
            is_hardcoded: bool = False
    ):
        """
        Initialize bandwidth object.

        Args:
            transferred_bytes: Data transferred in bytes (for measured mode)
            time: Transfer duration in seconds (for measured mode)
            hardcoded_value: Predefined bandwidth in Mbps (for hardcoded mode)
            is_hardcoded: Force hardcoded flag (optional)

        Raises:
            ValueError: If neither mode parameters are provided
        """
        # Mode 1: Hardcoded bandwidth (from config)
        if hardcoded_value is not None:
            # Convert Mbps to bytes/sec for internal storage
            self.bandwidth = (hardcoded_value * 1_000_000) / 8
            self.is_hardcoded = True

        # Mode 2: Measured bandwidth (from real transfer)
        elif transferred_bytes is not None and time is not None:
            if time <= 0:
                raise ValueError(f"Time must be positive, got {time}")
            if transferred_bytes < 0:
                raise ValueError(f"Transferred bytes cannot be negative, got {transferred_bytes}")

            self.bandwidth = transferred_bytes / time  # bytes/sec
            self.is_hardcoded = is_hardcoded

        # Error: No valid parameters
        else:
            raise ValueError(
                "Must provide either:\n"
                "  - hardcoded_value (Mbps), or\n"
                "  - (transferred_bytes, time) for measurement"
            )

    def __eq__(self, other):
        """Compare bandwidth values (mode-agnostic)"""
        if not isinstance(other, BandWidth):
            return False
        return abs(self.bandwidth - other.bandwidth) < 1e-6

    def __hash__(self):
        """Allow use as dict key"""
        return hash(round(self.bandwidth, 6))

    def __repr__(self):
        """Human-readable representation"""
        mode = "hardcoded" if self.is_hardcoded else "measured"
        return f"BandWidth({self.to_mbps():.2f} Mbps, {mode})"

    def to_mbps(self) -> float:
        """
        Convert to Mbps (Megabits per second).

        Returns:
            Bandwidth in Mbps
        """
        return (self.bandwidth * 8) / 1_000_000

    def to_mbytes_per_sec(self) -> float:
        """
        Convert to MB/s (Megabytes per second).

        Returns:
            Bandwidth in MB/s
        """
        return self.bandwidth / 1_000_000

    def to_bytes_per_sec(self) -> float:
        """
        Get raw bandwidth in bytes per second.

        Returns:
            Bandwidth in bytes/sec
        """
        return self.bandwidth

    def get_mode(self) -> str:
        """
        Get current bandwidth mode.

        Returns:
            'hardcoded' or 'measured'
        """
        return "hardcoded" if self.is_hardcoded else "measured"

    @staticmethod
    def from_mbps(mbps: float, is_hardcoded: bool = True) -> 'BandWidth':
        """
        Create BandWidth object from Mbps value.

        Args:
            mbps: Bandwidth in Mbps
            is_hardcoded: Mark as hardcoded (default: True)

        Returns:
            BandWidth instance
        """
        return BandWidth(hardcoded_value=mbps)

    @staticmethod
    def from_transfer(bytes_transferred: float, seconds: float) -> 'BandWidth':
        """
        Create BandWidth object from measured transfer.

        Args:
            bytes_transferred: Total bytes transferred
            seconds: Time elapsed in seconds

        Returns:
            BandWidth instance (measured mode)
        """
        return BandWidth(transferred_bytes=bytes_transferred, time=seconds)