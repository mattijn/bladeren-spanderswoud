"""Haal Sentinel-2 L2A (Earth Search, collection sentinel-2-c1-l2a, openbaar) op voor een venster rond Spanderswoud
en sla per opname NDVI, blauw en SCL op als kubus (s2/cube.npz).

Gebruik:  uv run --no-project --with rasterio,pyproj,numpy,requests python s2_extract.py
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
                  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")

HERE = Path(__file__).parent
OUT = HERE / "s2"
STAC = "https://earth-search.aws.element84.com/v1/search"
TILE = "31UFT"  # bevat Spanderswoud volledig
BBOX = (5.137, 52.239, 5.164, 52.257)
PERIOD = os.environ.get("S2_PERIOD", "2018-01-01T00:00:00Z/2026-11-02T23:59:59Z")
COLLECTION = os.environ.get("S2_COLLECTION", "sentinel-2-c1-l2a")  # 'sentinel-2-l2a' vult gaten (o.a. 2022)
CUBE = os.environ.get("S2_CUBE", "cube.npz")
MAX_TILE_CLOUD = 95  # ruim: lokaal kan het helder zijn terwijl de tegel grotendeels bewolkt is


def search():
    body = {"collections": [COLLECTION], "bbox": list(BBOX), "datetime": PERIOD, "limit": 200,
            "query": {"eo:cloud_cover": {"lt": MAX_TILE_CLOUD}, "mgrs:grid_square": {"eq": TILE[3:]}}}
    items, url = [], STAC
    while url:
        r = requests.post(url, json=body, timeout=120)
        r.raise_for_status()
        d = r.json()
        items += d["features"]
        nxt = [l for l in d.get("links", []) if l["rel"] == "next"]
        url, body = (nxt[0]["href"], nxt[0].get("body", body)) if nxt else (None, None)
    # dubbele producten (zelfde opname, ander processing-run) ontdubbelen op datum+tijd
    uniq = {}
    for f in items:
        uniq.setdefault(f["properties"]["datetime"][:19], f)
    return [uniq[k] for k in sorted(uniq)]


def utm_bounds():
    t = Transformer.from_crs(4326, 32631, always_xy=True)
    xs, ys = t.transform([BBOX[0], BBOX[2], BBOX[0], BBOX[2]], [BBOX[1], BBOX[1], BBOX[3], BBOX[3]])
    # uitlijnen op het 20 m-raster zodat 10 m en 20 m netjes samenvallen
    return (np.floor(min(xs) / 20) * 20, np.floor(min(ys) / 20) * 20,
            np.ceil(max(xs) / 20) * 20, np.ceil(max(ys) / 20) * 20)


def read(href, bounds, shape, resampling=Resampling.nearest):
    with rasterio.open(href) as src:
        win = from_bounds(*bounds, transform=src.transform)
        return src.read(1, window=win, out_shape=shape, resampling=resampling, boundless=True, fill_value=0)


def dn(item, band, bounds, shape):
    return read(item["assets"][band]["href"], bounds, shape).astype(np.float32)


def to_refl(item, band, d, has_offset):
    rb = item["assets"][band].get("raster:bands", [{}])[0]
    out = d * rb.get("scale", 1e-4) + (rb.get("offset", -0.1) if has_offset else 0.0)
    out[d == 0] = np.nan
    return out


def process(item, bounds, shape):
    try:
        raw = {b: dn(item, b, bounds, shape) for b in ("red", "nir", "blue")}
        # Vanaf processing baseline 04.00 bevatten DN's een offset van +1000. In 'sentinel-2-l2a' is die soms al
        # afgetrokken en is de vlag earthsearch:boa_offset_applied niet betrouwbaar; daarom uit de data bepalen:
        # rode reflectie boven bos/heide is ~0.02-0.05, dus met offset liggen vrijwel alle DN's boven 1000.
        valid = raw["red"][raw["red"] > 0]
        has_offset = valid.size > 0 and np.percentile(valid, 5) >= 1000
        red, nir, blue = (to_refl(item, b, raw[b], has_offset) for b in ("red", "nir", "blue"))
        scl = read(item["assets"]["scl"]["href"], bounds, shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.clip((nir - red) / (nir + red), -1, 1)
        return item["properties"]["datetime"][:10], ndvi.astype(np.float16), blue.astype(np.float16), scl
    except Exception as e:  # netwerkfout: opname overslaan, maar melden
        print("FOUT", item["id"], e)
        return None


def main():
    OUT.mkdir(exist_ok=True)
    items = search()
    print(len(items), "opnamen gevonden")
    bounds = utm_bounds()
    shape = (int((bounds[3] - bounds[1]) / 10), int((bounds[2] - bounds[0]) / 10))
    with ThreadPoolExecutor(8) as ex:
        res = [r for r in ex.map(lambda it: process(it, bounds, shape), items) if r is not None]
    res.sort(key=lambda r: r[0])
    np.savez_compressed(OUT / CUBE, dates=np.array([r[0] for r in res]),
                        ndvi=np.stack([r[1] for r in res]), blue=np.stack([r[2] for r in res]),
                        scl=np.stack([r[3] for r in res]))
    (OUT / CUBE.replace(".npz", "_grid.json")).write_text(json.dumps({"crs": "EPSG:32631", "bounds": bounds, "shape": shape, "res": 10,
                                               "items": [it["id"] for it in items]}))
    print("klaar:", len(res), "opnamen, venster", shape)


if __name__ == "__main__":
    main()
