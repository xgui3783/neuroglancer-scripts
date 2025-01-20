import math
import pathlib
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from numpy import ndarray

from neuroglancer_scripts.file_accessor import FileAccessor

from .base import Codec, TermCodecException

if TYPE_CHECKING:
    from ..metadata import Zarr3ArrayMetadata

@dataclass
class ShardingCodecCfg:
    chunk_shape: List[int]
    codecs: List[Codec]
    index_codecs: List[Codec]
    index_location: str

    @staticmethod
    def parse(configuration) -> "ShardingCodecCfg":
        assert isinstance(configuration, dict)

        chunk_shape = configuration.get("chunk_shape")
        assert isinstance(chunk_shape, list)
        assert all(isinstance(cs, int) for cs in chunk_shape)

        codecs = configuration.get("codecs")
        assert isinstance(codecs, list)

        index_codecs = configuration.get("index_codecs")
        assert isinstance(index_codecs, list)

        index_location = configuration.get("index_location", "end")
        assert index_location in ("start", "end")
        return ShardingCodecCfg(
            chunk_shape=chunk_shape,
            codecs=[Codec.parse(codec) for codec in codecs],
            index_codecs=[Codec.parse(ic) for ic in index_codecs],
            index_location=index_location,
        )


@dataclass
class ShardingCodec(Codec[ndarray, bytes]):
    configuration: ShardingCodecCfg

    _input = ndarray
    _output = bytes

    name: str = "sharding_indexed"

    def __post_init__(self):
        if isinstance(self.configuration, dict):
            self.configuration = ShardingCodecCfg(**self.configuration)

    def get_header_size(self, metadata: "Zarr3ArrayMetadata"):
        num_subchunk = math.prod(
            [chunk / subchunk for chunk, subchunk in zip(
                metadata.chunk_grid.configuration["chunk_shape"],
                self.configuration.chunk_shape
            )]
        )
        # TODO add support for crc checksum
        return int(num_subchunk * 16) # in bytes, assuming no crc checksum

    def get_subchunk_dim(self, metadata: "Zarr3ArrayMetadata"):
        return [
            chunk_grid / chunk_shape
            for chunk_grid, chunk_shape in zip(
                metadata.chunk_grid.configuration["chunk_shape"],
                self.configuration.chunk_shape
            )
        ]

    def get_chunk_coord_header_offset(self, chunk_coords, metadata: "Zarr3ArrayMetadata"):
        # TODO add support for header at end
        assert self.configuration.index_location == "start"

        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords

        subchunk_dim = self.get_subchunk_dim(metadata)

        relative_chunk_coord = [
            cc % cg
            for cc, cg in zip(
                [xmin, ymin, zmin],
                metadata.chunk_grid.configuration["chunk_shape"])]
        
        relative_grid_idx = [ rcc / cs
                             for rcc, cs in zip(
                                 relative_chunk_coord,
                                 self.configuration.chunk_shape)]
        return int(
            relative_grid_idx[2] * 16 +
            relative_grid_idx[1] * subchunk_dim[2] * 16 +
            relative_grid_idx[0] * subchunk_dim[2] * subchunk_dim[1] * 16
        )

    @classmethod
    def parse(cls, obj):
        assert isinstance(obj, dict)
        configuration = obj.get("configuration")
        assert isinstance(configuration, dict)

        return cls(configuration=ShardingCodecCfg.parse(configuration))

    def encode(self, input, metadata: "Zarr3ArrayMetadata", io, *args, path=None, chunk_coords=None, **kwargs):
        if chunk_coords is None:
            raise TypeError("chunk_coords is required by shardingcodec to encode chunk")
        if path is None:
            raise TypeError("path is required by shardingcodc to encode chunk")

        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        assert list(self.configuration.chunk_shape) == [xmax - xmin, ymax - ymin, zmax - zmin]

        accessor = io.accessor
        assert isinstance(accessor, FileAccessor)

        chunkcoord_hdroffset = self.get_chunk_coord_header_offset(chunk_coords, metadata)

        relative_path = pathlib.Path(path)
        foo = [int(v / y) for v, y in zip([xmin, ymin, zmin], self.configuration.chunk_shape)]
        print("foo", chunkcoord_hdroffset, foo, foo == [1, 0, 0])
        if foo == [1, 0, 0]:
            assert chunkcoord_hdroffset > 0
        file_path = accessor.base_path / relative_path
        file_path.parent.mkdir(exist_ok=True, parents=True)

        if not file_path.exists():
            file_path.write_bytes(b"\0" * self.get_header_size(metadata))

        output = input
        for codec in self.configuration.codecs:
            output = codec.encode(output, metadata, io, path=path, chunk_coords=chunk_coords)

        with file_path.open("r+b") as f:


            nbytes = len(output)
            offset = f.seek(0, 2)

            hdr_metadata = struct.pack("<QQ", offset, nbytes)
            f.write(output)

            f.seek(chunkcoord_hdroffset)
            f.write(hdr_metadata)

        raise TermCodecException("ShardCodec writes to io directly")


    def decode(self, output, metadata, io, *args, **kwargs):
        return super().decode(output, metadata)
