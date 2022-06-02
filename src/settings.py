import pathlib


class Config:
    DATA_DIR = "./data"
    RAW_DATA = pathlib.Path(f"{DATA_DIR}/raw")
    PROC_DATA = pathlib.Path(f"{DATA_DIR}/processed")
    TEST_DATA = pathlib.Path(f"{DATA_DIR}/test")
    CATALOGS = pathlib.Path("./catalogs.yaml")
    CL_SELECTION = pathlib.Path("./difred_params.yaml")
    PLOTDIR = pathlib.Path("./plots")
