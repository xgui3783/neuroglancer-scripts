import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from tqdm import tqdm

from neuroglancer_scripts.accessor import Accessor
from neuroglancer_scripts.precomputed_io import PrecomputedIO
from neuroglancer_scripts.volume_io.base_io import MultiResIOBase
from neuroglancer_scripts.volume_io.zarr3_io.codecs import TermCodecException
from neuroglancer_scripts.volume_io.zarr3_io.metadata import (
    Zarr3ArrayMetadata,
    Zarr3GroupMetadata,
)


class ZarrV3IO(MultiResIOBase):
    def __init__(self, accessor: Accessor):
        super().__init__()
        self.accessor = accessor

        group_metadata_json = json.loads(self.accessor.fetch_file("zarr.json"))
        self.group_metadata = Zarr3GroupMetadata(**group_metadata_json)

        assert len(self.group_metadata.attributes["ome"].multiscales) == 1
        multiscale = self.group_metadata.attributes["ome"].multiscales[0]
        self.array_ome_metadata = {
            ds.path: ds
            for ds in multiscale.datasets}

        with ThreadPoolExecutor() as ex:
            self.array_metadata = {
                path: Zarr3ArrayMetadata(**json.loads(b_array_metadata))
                for path, b_array_metadata in zip(
                    self.array_ome_metadata.keys(),
                    ex.map(
                        self.accessor.fetch_file,
                        [f"{p}/zarr.json" for p in self.array_ome_metadata.keys()]
                    ),
                )
            }


    def mirror_from(self, io: Any):
        assert self.accessor.can_write, "Cannot mirror when accessor is not writable"

        if isinstance(io, PrecomputedIO):

            for scale in io.info.get("scales", []):
                key = scale.get("key")
                print("processing", key)

                size = scale.get('size')

                chunk_sizes = scale.get('chunk_sizes')
                assert chunk_sizes, f"chunk_sizes not defined for scale: {key}"
                assert len(chunk_sizes) == 1, f"assert len(chunk_sizes) == 1, but got {len(chunk_sizes)}"
                chunk_size = chunk_sizes[0]
                assert len(chunk_size) == 3, f"assert len(chunk_size) == 3, but got {len(chunk_size)}"


                def mirror_chunk(idx):
                    z_chunk_idx, y_chunk_idx, x_chunk_idx = idx
                    chunk = io.read_chunk(key, (
                        x_chunk_idx * chunk_size[0], min((x_chunk_idx + 1) * chunk_size[0], size[0]),
                        y_chunk_idx * chunk_size[1], min((y_chunk_idx + 1) * chunk_size[1], size[1]),
                        z_chunk_idx * chunk_size[2], min((z_chunk_idx + 1) * chunk_size[2], size[2]),
                    ))
                    chunk = np.transpose(chunk)

                    if len(chunk.shape) > 4:
                        chunk = chunk.reshape(chunk.shape[:3])

                    chunk = np.pad(chunk, ( 
                        (0, chunk_size[0] - chunk.shape[0]),
                        (0, chunk_size[1] - chunk.shape[1]),
                        (0, chunk_size[2] - chunk.shape[2]),
                    ), "edge")

                    self.write_chunk(chunk, key, (
                        x_chunk_idx * chunk.shape[0], (x_chunk_idx + 1) * chunk.shape[0],
                        y_chunk_idx * chunk.shape[1], (y_chunk_idx + 1) * chunk.shape[1],
                        z_chunk_idx * chunk.shape[2], (z_chunk_idx + 1) * chunk.shape[2],
                    ))

                all_chunks = [
                    (z_chunk_idx, y_chunk_idx, x_chunk_idx)
                    for z_chunk_idx in range((size[2] - 1) // chunk_size[2] + 1)
                    for y_chunk_idx in range((size[1] - 1) // chunk_size[1] + 1)
                    for x_chunk_idx in range((size[0] - 1) // chunk_size[0] + 1)
                ]

                with ThreadPoolExecutor() as ex:
                    list(
                        tqdm(
                            ex.map(
                                mirror_chunk,
                                all_chunks,
                            ),
                            total=len(all_chunks),
                            leave=True
                        )
                    )
            return
        raise NotImplementedError(f"{io.__class__.__name__} cannot be mirrored")


    def write_chunk(self, chunk, scale_key, chunk_coords):
        assert scale_key in self.array_metadata
        array_metadata = self.array_metadata[scale_key]
        path = scale_key + "/" + array_metadata.format_path(chunk_coords)

        try:
            for codec in array_metadata.codecs:
                chunk = codec.encode(chunk, array_metadata, self, chunk_coords=chunk_coords, path=path)

            assert isinstance(chunk, bytes)
            self.accessor.store_file(path, chunk)
        except TermCodecException:
            pass

    def read_chunk(self, scale_key, chunk_coords):
        raise NotImplementedError

    @property
    def info(self):
        pass
