import gzip
from dataclasses import dataclass

from neuroglancer_scripts.volume_io.zarr3_io.codecs.base import Codec


@dataclass
class GzipCodecCfg:
    level: int

    @staticmethod
    def parse(configuration) -> "GzipCodecCfg":
        assert isinstance(configuration, dict)
        level = configuration.get("level")
        return GzipCodecCfg(level=level)


@dataclass
class GzipCodec(Codec[bytes, bytes]):
    configuration: GzipCodecCfg

    _input = bytes
    _output = bytes

    name: str = "gzip"

    def __post_init__(self):
        if isinstance(self.configuration, dict):
            self.configuration = GzipCodecCfg(**self.configuration)

    @classmethod
    def parse(cls, obj):
        return cls(configuration=GzipCodecCfg.parse(obj.get("configuration")))

    def encode(self, input, metadata, io, *args, **kwargs):
        assert isinstance(input, bytes), f"{type(input)} is not of type bytes"
        return gzip.compress(input, self.configuration.level)

    def decode(self, output, metadata, io, *args, **kwargs):
        assert isinstance(output, bytes)
        return gzip.decompress(output)
