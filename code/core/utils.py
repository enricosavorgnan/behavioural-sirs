import os
import yaml
import numpy as np

def expr(loader, node):
    """
    Take care of YAML expression
    """
    value = loader.construct_scalar(node)
    try:
        return eval(value, {"__builtins__": None}, {})
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{value}': {e}")


def save_simulation(simulation : np.ndarray | list[float], file_path : str):
    """
    Save a simulation in a given file path as a .CSV file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, simulation, delimiter=",")
    print("Simulation saved to", file_path)