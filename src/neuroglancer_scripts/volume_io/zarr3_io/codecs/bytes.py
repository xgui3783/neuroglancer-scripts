import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from neuroglancer_scripts.volume_io.zarr3_io.codecs.base import Codec

if TYPE_CHECKING:
    from neuroglancer_scripts.volume_io.zarr3_io.io import ZarrV3IO
    from neuroglancer_scripts.volume_io.zarr3_io.metadata import (
        Zarr3ArrayMetadata,
    )


@dataclass
class ByteCodecCfg:
    endian: str

    @staticmethod
    def parse(configuration) -> "ByteCodecCfg":
        assert isinstance(configuration, dict)
        endian = configuration.get("endian")
        if endian:
            assert endian in {"big", "little"}
        return ByteCodecCfg(endian=endian)

    def byteorder_equal(self, dtype: np.dtype):
        if dtype.byteorder == "=":
            return sys.byteorder == self.endian
        if dtype.byteorder == "<":
            return self.endian == "little"
        if dtype.byteorder == ">":
            return self.endian == "big"
        raise TypeError


@dataclass
class BytesCodec(Codec[np.ndarray, bytes]):
    configuration: ByteCodecCfg

    _input = np.ndarray
    _output = bytes

    name: str = "bytes"

    @classmethod
    def parse(cls, obj):
        return cls(configuration=ByteCodecCfg.parse(obj.get("configuration")))

    def encode(self, input, metadata: "Zarr3ArrayMetadata", io: "ZarrV3IO", *args, **kwargs):
        itemsize = input.dtype.itemsize
        if itemsize == 1:
            return input.tobytes("C")
        if not self.configuration.byteorder_equal(input):
            input = np.ascontiguousarray(
                input, dtype=input.dtype.newbyteorder("S")
            )
        return input.tobytes("C")

    def decode(self, output, metadata: "Zarr3ArrayMetadata", io: "ZarrV3IO", *args, **kwargs):
        dtype = np.dtype(metadata.data_type)
        if not self.configuration.byteorder_equal(dtype):
            dtype = dtype.newbyteorder("S")
        chunk_shape = metadata.chunk_grid.configuration["chunk_shape"]
        return np.frombuffer(output, dtype).reshape(chunk_shape)
