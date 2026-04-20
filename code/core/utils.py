import yaml

def expr(loader, node):
    """
    Take care of YAML expression
    """
    value = loader.construct_scalar(node)
    try:
        return eval(value, {"__builtins__": None}, {})
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{value}': {e}")