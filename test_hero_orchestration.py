import numpy as np
import rasterio
from rasterio.transform import from_origin
from src.orchestration.orchestrator import SATQueryOrchestrator

def make_dummy_tif(path):
    transform = from_origin(0, 0, 10, 10)
    meta = {
        'driver': 'GTiff',
        'height': 512,
        'width': 512,
        'count': 4,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': transform
    }
    with rasterio.open(path, 'w', **meta) as dst:
        dst.write(np.zeros((4, 512, 512), dtype='float32'))

def test():
    make_dummy_tif("S2_T1.tif")
    make_dummy_tif("S2_T2.tif")
    make_dummy_tif("S1_GRD.tif")
    
    orchestrator = SATQueryOrchestrator()
    inputs = ["S2_T1.tif", "S2_T2.tif", "S1_GRD.tif"]
    query = "Find newly constructed buildings within 500 m of flooded areas."
    
    print("Running orchestrator...")
    result = orchestrator.run(query=query, inputs=inputs)
    print("Success:", result.success)
    if not result.success:
        print("Messages:", result.messages)
        print("Metrics:", result.pipeline_metrics)
    
    if result.lifecycle_trace:
        resolution_step = [s for s in result.lifecycle_trace if s["stage"] == "INPUT_SCENE_RESOLUTION"]
        if resolution_step:
            print("Input Bindings:", resolution_step[0]["outputs"].get("input_bindings"))

if __name__ == "__main__":
    test()
