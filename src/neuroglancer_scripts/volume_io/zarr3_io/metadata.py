import json
import math
from abc import ABC
from dataclasses import asdict, dataclass
from typing import ClassVar, Dict, List, Type, Union

from neuroglancer_scripts.volume_io.zarr3_io.codecs import (
    ByteCodecCfg,
    BytesCodec,
    Codec,
    GzipCodec,
    GzipCodecCfg,
    ShardingCodec,
    ShardingCodecCfg,
)

VALID_DATATYPES = (
    "float16",
    "float32",
    "float64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    # "bool", "int8", "int16", "int32", "int64",
    # "complex64", "complex128",
    # "r*"
)


@dataclass
class ChunkGrid:
    name: str
    configuration: dict

    @staticmethod
    def parse(chunk_grid):
        assert isinstance(chunk_grid, dict)

        name = chunk_grid.get("name")
        assert name == "regular"

        configuration = chunk_grid.get("configuration")
        assert isinstance(configuration, dict)
        chunk_shape = configuration.get("chunk_shape")
        assert isinstance(chunk_shape, list)
        assert all(isinstance(cs, int) and cs > 0 for cs in chunk_shape)

        return ChunkGrid(name=name, configuration=configuration)


@dataclass
class ChunkKeyEncoding:
    pass


@dataclass
class DefaultChunkKeyEncoding(ChunkKeyEncoding):
    configuration: dict
    name: str = "default"

    @staticmethod
    def parse(chunk_key_encoding):
        assert isinstance(chunk_key_encoding, dict)

        name = chunk_key_encoding.get("name")
        assert name == "default"

        configuration = chunk_key_encoding.get("configuration", dict())
        assert isinstance(configuration, dict)

        separator = configuration.get("separator", "/")
        assert separator in ("/", ".")
        configuration["separator"] = separator

        return DefaultChunkKeyEncoding(name=name, configuration=configuration)


@dataclass
class Zarr3ArrayMetadata:
    shape: List[int]
    data_type: str
    chunk_grid: ChunkGrid
    chunk_key_encoding: DefaultChunkKeyEncoding
    fill_value: Union[float, int]
    codecs: List[Codec]

    # optional member, but since ngff requires it, might as well be required
    dimension_names: List[str]
    zarr_format: int = 3
    node_type: str = "array"


    def format_path(self, chunk_coords):
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        gridx, gridy, gridz = self.chunk_grid.configuration["chunk_shape"]
        separator = self.chunk_key_encoding.configuration["separator"]
        return f"c{separator}{xmin // gridx}{separator}{ymin // gridy}{separator}{zmin // gridz}"

    def __post_init__(self):

        assert isinstance(self.shape, list)
        assert all(isinstance(s, int) and s > 0 for s in self.shape)

        assert self.data_type in VALID_DATATYPES

        if isinstance(self.chunk_grid, dict):
            self.chunk_grid = ChunkGrid(**self.chunk_grid)

        if isinstance(self.chunk_key_encoding, dict):
            self.chunk_key_encoding = DefaultChunkKeyEncoding(**self.chunk_key_encoding)

        assert isinstance(self.fill_value, (int, float))

        assert isinstance(self.codecs, list)

        self.codecs = [Codec.parse(c) for c in self.codecs]


# following the metadata schema defined per
# ngff v0.5
# https://ngff.openmicroscopy.org/latest/


@dataclass
class Zarr3GroupAttrOmeAxis:
    name: str
    type: str  # time|channel
    unit: str


class Zarr3GroupAttrOmeXform(ABC):
    type: str
    _xform_registry: ClassVar[Dict[str, Type["Zarr3GroupAttrOmeXform"]]] = {}

    def __init_subclass__(cls):
        assert cls.type not in cls._xform_registry
        cls._xform_registry[cls.type] = cls
        return super().__init_subclass__()


@dataclass
class Zarr3GroupAttrOmeXformId(Zarr3GroupAttrOmeXform):
    type: str = "identity"


@dataclass
class Zarr3GroupAttrOmeXformTransl(Zarr3GroupAttrOmeXform):
    translation: List[float]

    type: str = "translation"
    path: str = None  # NYI, allowed by spec


@dataclass
class Zarr3GroupAttrOmeXformScale(Zarr3GroupAttrOmeXform):
    scale: List[float]

    type: str = "scale"
    path: str = None  # NYI, allowed by spec


@dataclass
class Zarr3GroupAttrOmeDataset:
    path: str
    coordinateTransformations: List[Zarr3GroupAttrOmeXform]

    def validate(self):

        assert isinstance(self.coordinateTransformations, list)
        assert all(
            xform.type in {"translation", "scale"}
            for xform in self.coordinateTransformations
        )

        scale_xforms = [
            xform
            for xform in self.coordinateTransformations
            if xform.type == "scale"
        ]
        transl_xforms = [
            xform
            for xform in self.coordinateTransformations
            if xform.type == "translation"
        ]
        assert len(scale_xforms) == 1
        assert len(transl_xforms) <= 1

        scale_indices = map(self.coordinateTransformations.index, scale_xforms)
        transl_indices = map(
            self.coordinateTransformations.index, transl_xforms
        )
        assert all(idx < min(scale_indices) for idx in transl_indices)


@dataclass
class Zarr3GroupAttrOmeScale:
    axes: List[Zarr3GroupAttrOmeAxis]
    datasets: List[Zarr3GroupAttrOmeDataset]

    coordinateTransformations: List[Zarr3GroupAttrOmeXform] = None
    name: str = None
    type: str = None
    metadata: Dict = None
    metadata: Dict = None

    def __post_init__(self):
        self.axes = [
            axis if isinstance(axis, Zarr3GroupAttrOmeAxis) else Zarr3GroupAttrOmeAxis(**axis)
            for axis in self.axes]

        self.datasets = [
            ds if isinstance(ds, Zarr3GroupAttrOmeDataset) else Zarr3GroupAttrOmeDataset(**ds)
            for ds in self.datasets
        ]

        if not self.coordinateTransformations:
            self.coordinateTransformations = []
        self.coordinateTransformations = [
            Zarr3GroupAttrOmeXform(**xform)
            for xform in self.coordinateTransformations]


@dataclass
class Zarr3GroupAttrOme:
    multiscales: List[Zarr3GroupAttrOmeScale]
    version: str = "0.5"

    def __post_init__(self):
        self.multiscales = [
            s if isinstance(s, Zarr3GroupAttrOmeScale) else Zarr3GroupAttrOmeScale(**s)
            for s in self.multiscales]


@dataclass
class Zarr3GroupMetadata:
    attributes: Dict[str, Zarr3GroupAttrOme]
    zarr_format: int = 3
    node_type: str = "group"

    def __post_init__(self):
        if (
            self.attributes
            and "ome" in self.attributes
            and not isinstance(self.attributes["ome"], Zarr3GroupAttrOme)
        ):

            self.attributes["ome"] = Zarr3GroupAttrOme(**self.attributes["ome"])


def from_precomputed_info(info):
    assert isinstance(info, dict)
    data_type = info.get("data_type")
    num_channels = info.get("num_channels")
    type = info.get("type")

    assert type == "image"
    assert num_channels == 1

    scales = info.get("scales")
    assert isinstance(scales, list)

    scale0 = scales[0]
    assert isinstance(scale0, dict)
    resolution = scale0.get("resolution")
    size = scale0.get("size")

    group_metadata = Zarr3GroupMetadata(attributes={
        "ome": Zarr3GroupAttrOme(
            multiscales=[
                Zarr3GroupAttrOmeScale(
                    axes=[
                        Zarr3GroupAttrOmeAxis(name="x", type="space", unit="nanometer"),
                        Zarr3GroupAttrOmeAxis(name="y", type="space", unit="nanometer"),
                        Zarr3GroupAttrOmeAxis(name="z", type="space", unit="nanometer"),
                    ],
                    datasets=[
                        Zarr3GroupAttrOmeDataset(
                            path=scale.get("key"),
                            coordinateTransformations=[
                                Zarr3GroupAttrOmeXformScale(scale=scale.get("resolution"))
                            ])
                        for scale in scales
                    ])
            ])
    })

    yield "zarr.json", json.dumps(
        asdict(group_metadata),
        indent=2
    ).encode("utf-8")

    for scale in scales:
        shape = scale.get("size")
        chunk_shape = scale.get("chunk_sizes")[0]

        assert len(shape) == len(chunk_shape) == 3

        # do not exceed 4 chunks in each dimension
        # which leads to max 64 files per scale

        chunk_merge: List[int] = [math.ceil(s / cs / 4) for s, cs in zip(shape, chunk_shape)]
        array_metadata = Zarr3ArrayMetadata(
            shape=shape,
            data_type=data_type,
            chunk_grid=ChunkGrid(name="regular", configuration={
                "chunk_shape": [cs * chm for cs, chm in zip(chunk_shape, chunk_merge)]
            }),
            chunk_key_encoding=DefaultChunkKeyEncoding(configuration={
                "separator": "/"
            }),
            dimension_names=["x", "y", "z" ],
            fill_value=0,
            codecs=[
                ShardingCodec(
                    configuration=ShardingCodecCfg(
                        chunk_shape=chunk_shape,
                        codecs=[
                            BytesCodec(
                                configuration=ByteCodecCfg(endian="little")
                            ),
                            GzipCodec(
                                configuration=GzipCodecCfg(9)
                            )
                        ],
                        index_location="start",
                        index_codecs=[
                            BytesCodec(
                                configuration=ByteCodecCfg(endian="little")
                            )
                        ]
                    )
                ),
            ]
        )
        yield scale.get("key") + "/zarr.json", json.dumps(
            asdict(array_metadata),
            indent=2
        ).encode("utf-8")
