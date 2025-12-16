import sys
from gr00t.experiment.data_config import load_data_config
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING

def inspect_config():
    config_name = "so100_dualcam"
    try:
        print(f"Loading config: {config_name}")
        data_config_cls = load_data_config(config_name)
        
        print(f"Class: {type(data_config_cls).__name__}")
        
        modality = data_config_cls.modality_config()
        print("\nModality Config:")
        for key, val in modality.items():
            print(f"  {key}: {val}")
            
        transforms = data_config_cls.transform()
        print(f"\nTransforms: {transforms}")
        
        if hasattr(data_config_cls, "action_indices"):
            print(f"\nAction Indices (Horizon): {len(data_config_cls.action_indices)}")
            
        print("\nEmbodiment Tag Mapping keys:")
        print(list(EMBODIMENT_TAG_MAPPING.keys()))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_config()

