import importlib


def get_instructions(instructions: str | None) -> str | None:
    if instructions and instructions.startswith("import::"):
        _, full_path = instructions.lstrip("import::")
        module, obj = full_path.rsplit(".", 1)
        module = importlib.import_module(module)
        return getattr(module, obj)
    return instructions
