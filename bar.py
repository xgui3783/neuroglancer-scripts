import json

from neuroglancer_scripts.volume_io.zarr3_io import ZarrV3IO, from_precomputed_info
from neuroglancer_scripts.file_accessor import FileAccessor
from neuroglancer_scripts.http_accessor import HttpAccessor
from neuroglancer_scripts.precomputed_io import PrecomputedIO

waxholm_url = "https://neuroglancer.humanbrainproject.eu/precomputed/WHS_SD_rat/templates/v1.01/t2star_masked"
bigbrain_url = "https://neuroglancer.humanbrainproject.eu/precomputed/BigBrainRelease.2015/8bit"

src_http = HttpAccessor(base_url=bigbrain_url)

src_info = json.loads(src_http.fetch_file("info"))

src_io = PrecomputedIO(
    src_info,
    src_http)

dst_acc = FileAccessor("./test_data/bigbrain", gzip=False)

for filename, filebytes in from_precomputed_info(src_info):
    dst_acc.store_file(filename, filebytes)



def iter_chunks(info):

    for scale in info.get("scales", []):
        key = scale.get('key')
        assert key, f"key not defined"
        
        size = scale.get('size')
        assert size, f"size not defined for scale: {key}"
        assert len(size) == 3
        
        chunk_sizes = scale.get('chunk_sizes')
        assert chunk_sizes, f"chunk_sizes not defined for scale: {key}"
        assert len(chunk_sizes) == 1, f"assert len(chunk_sizes) == 1, but got {len(chunk_sizes)}"
        chunk_size = chunk_sizes[0]
        assert len(chunk_size) == 3, f"assert len(chunk_size) == 3, but got {len(chunk_size)}"
        for z_chunk_idx in range((size[2] - 1) // chunk_size[2] + 1):
            for y_chunk_idx in range((size[1] - 1) // chunk_size[1] + 1):
                for x_chunk_idx in range((size[0] - 1) // chunk_size[0] + 1):
                    yield key, (
                        x_chunk_idx * chunk_size[0], min((x_chunk_idx + 1) * chunk_size[0], size[0]),
                        y_chunk_idx * chunk_size[1], min((y_chunk_idx + 1) * chunk_size[1], size[1]),
                        z_chunk_idx * chunk_size[2], min((z_chunk_idx + 1) * chunk_size[2], size[2]),
                    )

dest_io = ZarrV3IO(dst_acc)

dest_io.mirror_from(src_io)

"""
https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B0.00002116666666666667%2C%22m%22%5D%2C%22y%22:%5B0.00002%2C%22m%22%5D%2C%22z%22:%5B0.00002116666666666667%2C%22m%22%5D%7D%2C%22position%22:%5B2407.791259765625%2C3934.491943359375%2C2944%5D%2C%22crossSectionScale%22:4.481689070338065%2C%22projectionScale%22:8192%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22https://data-proxy.ebrains.eu/api/v1/buckets/reference-atlas-data/zarr3/BigBrainRelease.2015/8bit/%7Czarr3:%22%2C%22tab%22:%22source%22%2C%22name%22:%228bit%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%228bit%22%7D%2C%22layout%22:%224panel%22%7D

Error retrieving chunk [object Object]:12,15,7: Error: Failed to decode gzip
Error retrieving chunk [object Object]:11,16,12: Error: Failed to decode gzip
Error retrieving chunk [object Object]:15,14,9: Error: Failed to decode gzip
Error retrieving chunk [object Object]:7,13,9: Error: Failed to decode gzip

looks like it's not deterministic. The gzip decoding fails in different grid coords

"""