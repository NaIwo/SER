import importlib


def get_dataset_by_name(dataset_name: str):
    module = importlib.import_module(f"src.datasets.datasets.{dataset_name.lower()}.{dataset_name.lower()}_dataset")
    class_name = f'{dataset_name.capitalize()}Dataset'
    return getattr(module, class_name)
