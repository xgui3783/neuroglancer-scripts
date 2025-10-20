import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from tqdm import tqdm

from neuroglancer_scripts.accessor import Accessor
from neuroglancer_scripts.precomputed_io import PrecomputedIO
from neuroglancer_scripts.volume_io.base_io import MultiResIOBase
from neuroglancer_scripts.volume_io.zarr3_io.codecs import TermCodecError
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
                        [f"{p}/zarr.json"
                         for p in self.array_ome_metadata.keys()]
                    ),
                )
            }


    def mirror_from(self, io: Any):
        assert self.accessor.can_write, "Accessor needs to be writable"

        if isinstance(io, PrecomputedIO):

            for scale in io.info.get("scales", []):
                key = scale.get("key")
                print("processing", key)

                size = scale.get('size')

                chszs = scale.get('chunk_sizes')
                assert chszs, f"chunk_sizes not defined for scale: {key}"
                assert len(chszs) == 1, f"assert {len(chszs)=} == 1"
                chsz = chszs[0]
                assert len(chsz) == 3, f"assert {len(chsz)=} == 3"


                def mirror_chunk(idx: tuple[int, int, int]):
                    z_i, y_i, x_i = idx
                    chunk = io.read_chunk(key, (
                        x_i * chsz[0], min((x_i + 1) * chsz[0], size[0]),
                        y_i * chsz[1], min((y_i + 1) * chsz[1], size[1]),
                        z_i * chsz[2], min((z_i + 1) * chsz[2], size[2]),
                    ))
                    chunk = np.transpose(chunk)

                    if len(chunk.shape) > 3:
                        chunk = chunk.reshape(chunk.shape[:3])

                    chunk = np.pad(chunk, (
                        (0, chsz[0] - chunk.shape[0]),
                        (0, chsz[1] - chunk.shape[1]),
                        (0, chsz[2] - chunk.shape[2]),
                    ), "edge")

                    self.write_chunk(chunk, key, (
                        x_i * chunk.shape[0], (x_i + 1) * chunk.shape[0],
                        y_i * chunk.shape[1], (y_i + 1) * chunk.shape[1],
                        z_i * chunk.shape[2], (z_i + 1) * chunk.shape[2],
                    ))

                all_chunks = [
                    (z_i, y_i, x_i)
                    for z_i in range((size[2] - 1) // chsz[2] + 1)
                    for y_i in range((size[1] - 1) // chsz[1] + 1)
                    for x_i in range((size[0] - 1) // chsz[0] + 1)
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
        raise NotImplementedError(f"{io.__class__.__name__} NYI")


    def write_chunk(self, chunk, scale_key, chunk_coords):
        assert scale_key in self.array_metadata
        array_metadata = self.array_metadata[scale_key]
        path = scale_key + "/" + array_metadata.format_path(chunk_coords)

        try:
            for codec in array_metadata.codecs:
                chunk = codec.encode(chunk, array_metadata, self,
                                     chunk_coords=chunk_coords, path=path)

            assert isinstance(chunk, bytes)
            self.accessor.store_file(path, chunk)
        except TermCodecError:
            pass

    def read_chunk(self, scale_key, chunk_coords):
        raise NotImplementedError

    @property
    def info(self):
        pass
