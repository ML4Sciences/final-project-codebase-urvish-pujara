import numpy as np
import importlib.util

def check_important_features(input_data):
    # Load important features from features.py
    spec = importlib.util.spec_from_file_location("reduced_features", "reduced_features.py")
    features_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features_module)
    important_features = features_module.important_features

    # Get the features in the input data
    input_features = list(input_data.keys())
    
    # Check if all important features are present in the input data
    missing_features = set(important_features) - set(input_features)
    if missing_features:
        print("ERROR: The following important features are missing from the input data:")
        print(list(missing_features))
        return False

    # Check if there are any extra features in the input data that are not important
    extra_features = set(input_features) - set(important_features)
    if extra_features:
        print("WARNING: The following extra features are present in the input data, but they are not important:")
        print(list(extra_features))
    
    # All important features are present in the input data
    return True
