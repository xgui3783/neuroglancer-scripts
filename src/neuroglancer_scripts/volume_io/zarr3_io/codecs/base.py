from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Generic, TypeVar

if TYPE_CHECKING:
    from ..io import ZarrV3IO
    from ..metadata import Zarr3ArrayMetadata

I = TypeVar("I")
O = TypeVar("O")


class Codec(Generic[I, O], ABC):
    _codec_registry: Dict[str, "Codec"] = {}
    _input: str
    _output: str
    name: str
    configuration: Any

    def __init_subclass__(cls):
        assert cls.name not in cls._codec_registry
        cls._codec_registry[cls.name] = cls

    @classmethod
    def parse(cls, obj) -> "Codec":
        if isinstance(obj, Codec):
            return obj
        assert isinstance(obj, dict)
        name = obj.get("name")
        assert (
            name in cls._codec_registry
        ), f"{name} not in codec registry. Not supported yet."

        return cls._codec_registry[name].parse(obj)

    # some array -> byte codec may need additional info (e.g. grid coord/ chunk coord of the current array being written)
    def encode(self, input, metadata: "Zarr3ArrayMetadata", io: "ZarrV3IO", *args, **kwargs):
        raise NotImplementedError

    def decode(self, output, metadata: "Zarr3ArrayMetadata", io: "ZarrV3IO", *args, **kwargs):
        raise NotImplementedError

class CodecException(Exception): pass

class InvalidCfgCodecException(Exception): pass

class TermCodecException(CodecException): pass
