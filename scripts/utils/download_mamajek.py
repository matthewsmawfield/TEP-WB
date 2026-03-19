from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.downloader import smart_download

URL = "https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt"


def download_mamajek(target_path=None):
    target = Path(target_path) if target_path else PROJECT_ROOT / "data" / "raw" / "EEM_dwarf_UBVIJHK_colors_Teff.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    return smart_download(URL, target, min_size_mb=0.05, n_workers=1)


if __name__ == "__main__":
    download_mamajek()
