import pathlib


class Config:
    DATA_DIR = "./data"
    RAW_DATA = pathlib.Path(f"{DATA_DIR}/raw")
    PROC_DATA = pathlib.Path(f"{DATA_DIR}/processed")
    CATALOGS = pathlib.Path("./catalogs.yaml")
