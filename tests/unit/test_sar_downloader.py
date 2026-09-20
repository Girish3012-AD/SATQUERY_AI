"""
Focused unit tests for the Sentinel-1 SAR windowed extraction data path.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rasuwa_sar_e2e import run_rasuwa_sar_pipeline, _download_aoi_window

@patch("rasuwa_sar_e2e._download_aoi_window")
@patch("src.data.sentinel1_stac.Sentinel1STACDiscovery.search_best")
def test_sar_e2e_pipeline(mock_search_best, mock_download, tmp_path):
    mock_scene = MagicMock()
    mock_scene.item_id = "test_s1_scene"
    mock_scene.datetime = "2024-10-15T00:00:00Z"
    mock_scene.collection = "sentinel-1-rtc"
    mock_scene.assets = {"VV": "https://test.url/vv.tif"}
    mock_scene.to_dict.return_value = {}
    mock_search_best.return_value = mock_scene

    def mock_extract(href, aoi_bbox, dest):
        # Create a dummy raster to satisfy SARSpecialist
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = np.ones((10, 10), dtype=np.float32)
        transform = from_origin(85.0, 28.5, 10.0, 10.0)
        
        with rasterio.open(
            dest, 'w', driver='GTiff',
            height=10, width=10,
            count=1, dtype=data.dtype,
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        return dest

    mock_download.side_effect = mock_extract

    with patch("rasuwa_sar_e2e.OUTPUT_DIR", tmp_path):
        with patch("rasuwa_sar_e2e.AUDIT_FILE", tmp_path / "audit.json"):
            audit = run_rasuwa_sar_pipeline()
    
    assert "aoi" in audit
    assert "scene" in audit
    assert "evidence" in audit
    assert "verification" in audit
    
    assert audit["verification"]["status"] == "verified"
    assert audit["evidence"]["task"] == "sar_analysis"
    assert audit["evidence"]["modality"] == "sar"
