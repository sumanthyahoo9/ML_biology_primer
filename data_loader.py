import subprocess
from pathlib import Path

REPO_URL = "https://github.com/alchemab/antiberta.git"
DATA_DIR = Path("data/antiberta_repo")


def clone_repo():
    if DATA_DIR.exists():
        return DATA_DIR
    cmd = ["git", "clone", "--depth", "1", REPO_URL, str(DATA_DIR)]
    subprocess.run(cmd, check=True)
    return DATA_DIR


def list_asset_files(pattern="*"):
    assets = DATA_DIR / "assets"
    return sorted(assets.rglob(pattern))


def list_paratope_files():
    return [f for f in list_asset_files("*.parquet") if "sabdab" in f.name.lower()]


if __name__ == "__main__":
    clone_repo()
    print("Paratope (SAbDab) files:", list_paratope_files())