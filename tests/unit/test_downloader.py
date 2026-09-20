"""
Unit tests for the resilient _download_aoi_window windowed extraction mechanism.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import rasterio
from rasterio.windows import Window
from affine import Affine

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rasuwa_flood_e2e import _download_aoi_window

@pytest.fixture
def mock_sign_href():
    with patch("rasuwa_flood_e2e._sign_href", return_value="https://signed.url") as mock_sign:
        yield mock_sign

@pytest.fixture
def mock_rasterio_open():
    mock_src = MagicMock()
    mock_src.crs = rasterio.crs.CRS.from_epsg(32645) # Example UTM zone
    mock_src.transform = Affine(10.0, 0.0, 300000.0, 0.0, -10.0, 4000000.0)
    
    # Return a dummy 10x10 window of ones
    mock_src.read.return_value = np.ones((10, 10), dtype=np.uint16)
    mock_src.window_transform.return_value = Affine(10.0, 0.0, 300500.0, 0.0, -10.0, 3999500.0)
    
    mock_src.meta = {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": 0,
        "width": 10980,
        "height": 10980,
        "count": 1,
        "crs": rasterio.crs.CRS.from_epsg(32645),
        "transform": mock_src.transform
    }
    
    # Enter/exit context manager
    mock_src.__enter__.return_value = mock_src
    
    with patch("rasterio.open") as mock_open:
        # rasterio.open is called twice: once for reading, once for writing.
        # We need the writing one to just be a dummy context manager too.
        mock_dst = MagicMock()
        mock_dst.__enter__.return_value = mock_dst
        
        def side_effect(fp, mode="r", **kwargs):
            if mode == "r":
                return mock_src
            return mock_dst
            
        mock_open.side_effect = side_effect
        yield mock_open, mock_src, mock_dst

@patch("rasterio.warp.transform_bounds")
def test_windowed_extraction(mock_transform_bounds, tmp_path, mock_sign_href, mock_rasterio_open):
    mock_open, mock_src, mock_dst = mock_rasterio_open
    
    # Mock the output of transform_bounds to return coordinates that map to a 10x10 window
    # left, bottom, right, top
    mock_transform_bounds.return_value = (300500.0, 3999400.0, 300600.0, 3999500.0)
    
    dest = tmp_path / "test_window.tif"
    aoi = [85.0, 28.0, 85.5, 28.5]
    
    _download_aoi_window("dummy_href", aoi, dest)
    
    mock_sign_href.assert_called_once_with("dummy_href")
    mock_transform_bounds.assert_called_once_with("EPSG:4326", mock_src.crs, *aoi)
    
    # Verify src.read was called with a window
    mock_src.read.assert_called_once()
    kwargs = mock_src.read.call_args[1]
    assert "window" in kwargs
    
    # Verify dst.write was called with data
    mock_dst.write.assert_called_once()
    args, kwargs = mock_dst.write.call_args
    data = args[0]
    assert data.shape == (10, 10)
    assert np.all(data == 1)
