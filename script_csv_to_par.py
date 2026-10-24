import pandas as pd
from pathlib import Path

DATA = Path("data/raw")
OUT = Path("data_par/raw")
OUT.mkdir(exist_ok=True)


# Function to fix Mojibake (garbled text)
def fix_mojibake(text):
    if isinstance(text, str):
        try:
            # Try reverse conversion: string -> cp1251 -> utf-8
            return text.encode('cp1251').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            # If it fails (e.g., text is already correct or it's Latin),
            # return the original string without changes
            return text
    return text


# 1. View log
VIEW_COLS = ["user_id", "vsetv_id", "start_time", "stop_time", "duraton"]

views = pd.read_csv(
    DATA / "dataset11-30.csv",
    dtype={
        "user_id": str,
        "vsetv_id": str,
        "start_time": str,
        "stop_time": str,
        "duraton": "float64",
    },
    keep_default_na=False,
    na_values=[""],
)
views.columns = views.columns.str.strip()
assert list(views.columns) == VIEW_COLS, views.columns.tolist()

views.to_parquet(OUT / "views.parquet", index=False)
print("views:", len(views))
del views


# 2. Three schedule files
SCH_COLS = [
    "channel_id", "channel_title", "start_time", "duration",
    "tv_show_title", "tv_show_id", "tv_show_category",
    "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
    "year_of_production", "director", "actors",
]

# Columns that may contain Russian text and need to be fixed
TEXT_COLS_TO_FIX = [
    "channel_title", "tv_show_title", "tv_show_category", 
    "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", 
    "director", "actors"
]

for name in ["export_arh_11-20-final", "export_arh_21-30-final", "export_arh_31-42-final"]:
    sch = pd.read_csv(
        DATA / f"{name}.csv",
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )
    sch.columns = sch.columns.str.strip()
    assert list(sch.columns) == SCH_COLS, (name, sch.columns.tolist())

    # Apply the fix function to text columns
    for col in TEXT_COLS_TO_FIX:
        sch[col] = sch[col].apply(fix_mojibake)

    sch.to_parquet(OUT / f"{name}.parquet", index=False)
    print(f"{name} fixed and saved. Rows: {len(sch)}")
    del sch



