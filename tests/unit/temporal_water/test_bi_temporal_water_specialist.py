import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

from src.executor.bi_temporal_water_specialist import BiTemporalWaterChangeSpecialist


@pytest.fixture
def create_synthetic_raster(tmp_path):
    """Factory fixture to create synthetic Sentinel-2 style 4-band rasters."""
    def _create(name: str, water_mask: np.ndarray, resolution_m: float = 10.0) -> Path:
        filepath = tmp_path / f"{name}.tif"
        
        # 4 bands: B02, B03, B04, B08. We care about B03 (idx 2) and B08 (idx 4)
        # Using 1-based indexing for rasterio, so green=2, nir=4 in our test setup?
        # Actually BiTemporalWaterChangeSpecialist uses DEFAULT_GREEN_BAND=1, DEFAULT_NIR_BAND=4
        # Let's write a 4-band image.
        height, width = water_mask.shape
        
        # Water: Green > NIR (NDWI > 0)
        # Non-water: Green < NIR (NDWI < 0)
        
        green_band = np.where(water_mask, 1500, 500).astype(np.float32)
        nir_band = np.where(water_mask, 500, 1500).astype(np.float32)
        
        transform = from_origin(300000.0, 3100000.0, resolution_m, resolution_m)
        
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=4,
            dtype='float32',
            crs='EPSG:32645',
            transform=transform,
        ) as dst:
            dst.write(np.zeros_like(green_band), 1) # B02
            dst.write(green_band, 2)                # B03
            dst.write(np.zeros_like(green_band), 3) # B04
            dst.write(nir_band, 4)                  # B08
            
        return filepath
    return _create


def test_bi_temporal_water_gain(create_synthetic_raster):
    """Test water expansion (gain) scenario."""
    # 10x10 grid (100 pixels). 10m resolution = 100m^2 per pixel.
    # T1: 2 water pixels = 200 m2
    t1_mask = np.zeros((10, 10), dtype=bool)
    t1_mask[0, 0] = True
    t1_mask[0, 1] = True
    
    # T2: 4 water pixels = 400 m2
    t2_mask = np.zeros((10, 10), dtype=bool)
    t2_mask[0, 0] = True
    t2_mask[0, 1] = True
    t2_mask[1, 0] = True
    t2_mask[1, 1] = True
    
    t1_path = create_synthetic_raster("t1_gain", t1_mask, 10.0)
    t2_path = create_synthetic_raster("t2_gain", t2_mask, 10.0)
    
    specialist = BiTemporalWaterChangeSpecialist(min_area_px=1)
    # Override default bands for test fixture
    params = {"green_band": 2, "nir_band": 4, "ndwi_threshold": 0.0}
    
    evidence = specialist.infer([str(t1_path), str(t2_path)], parameters=params)
    
    m = evidence.measurement
    assert m["area_t1_m2"] == 200.0
    assert m["area_t2_m2"] == 400.0
    assert m["absolute_change_m2"] == 200.0
    assert m["percentage_change"] == 100.0
    assert m["water_gain_m2"] == 200.0
    assert m["water_loss_m2"] == 0.0
    
    assert "establish flooding" in evidence.result["note"]
    assert evidence.result["label"] == "spectral_water_area_change"


def test_bi_temporal_water_loss(create_synthetic_raster):
    """Test water contraction (loss) scenario."""
    # T1: 5 pixels = 500 m2
    t1_mask = np.zeros((10, 10), dtype=bool)
    t1_mask[0:5, 0] = True
    
    # T2: 1 pixel = 100 m2
    t2_mask = np.zeros((10, 10), dtype=bool)
    t2_mask[0, 0] = True
    
    t1_path = create_synthetic_raster("t1_loss", t1_mask, 10.0)
    t2_path = create_synthetic_raster("t2_loss", t2_mask, 10.0)
    
    specialist = BiTemporalWaterChangeSpecialist(min_area_px=1)
    evidence = specialist.infer(
        [str(t1_path), str(t2_path)], 
        parameters={"green_band": 2, "nir_band": 4}
    )
    
    m = evidence.measurement
    assert m["area_t1_m2"] == 500.0
    assert m["area_t2_m2"] == 100.0
    assert m["absolute_change_m2"] == -400.0
    assert m["percentage_change"] == -80.0
    assert m["water_gain_m2"] == 0.0
    assert m["water_loss_m2"] == 400.0


def test_bi_temporal_zero_area_t1(create_synthetic_raster):
    """Test handling of 0 area in T1 (prevent division by zero)."""
    t1_mask = np.zeros((10, 10), dtype=bool)
    t2_mask = np.zeros((10, 10), dtype=bool)
    t2_mask[0, 0] = True
    
    t1_path = create_synthetic_raster("t1_zero", t1_mask, 10.0)
    t2_path = create_synthetic_raster("t2_zero", t2_mask, 10.0)
    
    specialist = BiTemporalWaterChangeSpecialist(min_area_px=1)
    evidence = specialist.infer(
        [str(t1_path), str(t2_path)], 
        parameters={"green_band": 2, "nir_band": 4}
    )
    
    m = evidence.measurement
    assert m["area_t1_m2"] == 0.0
    assert m["area_t2_m2"] == 100.0
    assert m["absolute_change_m2"] == 100.0
    assert m["percentage_change"] == float("inf")
    assert m["water_gain_m2"] == 100.0


def test_bi_temporal_provenance_and_metadata(create_synthetic_raster):
    """Test that CRS, resolution, and input parameters are preserved."""
    t1_mask = np.zeros((10, 10), dtype=bool)
    t2_mask = np.zeros((10, 10), dtype=bool)
    
    t1_path = create_synthetic_raster("t1_prov", t1_mask, 20.0) # 20m resolution!
    t2_path = create_synthetic_raster("t2_prov", t2_mask, 20.0)
    
    specialist = BiTemporalWaterChangeSpecialist(min_area_px=1)
    evidence = specialist.infer(
        [str(t1_path), str(t2_path)], 
        parameters={
            "green_band": 2, 
            "nir_band": 4, 
            "ndwi_threshold": 0.1,
            "scene_id_t1": "SCENE_A",
            "scene_id_t2": "SCENE_B",
        }
    )
    
    assert evidence.measurement["resolution_m"] == (20.0, 20.0)
    assert evidence.measurement["crs"] == "EPSG:32645"
    assert evidence.measurement["ndwi_threshold"] == 0.1
    
    assert evidence.result["t1_scene_id"] == "SCENE_A"
    assert evidence.result["t2_scene_id"] == "SCENE_B"
    
    assert "t1_provenance" in evidence.provenance
    assert "t2_provenance" in evidence.provenance
    
    # Must NOT claim flood confirmed
    assert "flood confirmed" not in evidence.result["note"].lower()
    assert "spectral water-area change" in evidence.result["note"].lower()
