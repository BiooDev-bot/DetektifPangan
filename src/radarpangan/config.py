"""Memuat config.yaml dan menyiapkan path project.

Dipakai oleh semua notebook & script:

    from radarpangan.config import load_project
    cfg, P = load_project()
    P.interim / "pihps_long.parquet"
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def find_project_root(start: str | Path | None = None) -> Path:
    """Cari folder project = folder terdekat (ke atas) yang berisi config.yaml."""
    here = Path(start or Path.cwd()).resolve()
    for folder in [here, *here.parents]:
        if (folder / "config.yaml").exists():
            return folder
    raise FileNotFoundError(
        "config.yaml tidak ditemukan. Jalankan dari dalam folder MiningProcess "
        "(atau subfoldernya, mis. notebooks/)."
    )


@dataclass
class Paths:
    root: Path
    raw_pihps: Path
    raw_weather: Path
    external: Path
    interim: Path
    processed: Path
    output: Path
    models: Path
    figures: Path
    tables: Path
    site: Path

    def makedirs(self) -> None:
        for name, value in self.__dict__.items():
            if name != "root":
                Path(value).mkdir(parents=True, exist_ok=True)


def load_project(start: str | Path | None = None) -> tuple[dict, Paths]:
    root = find_project_root(start)
    with open(root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    paths = Paths(root=root, **{k: root / v for k, v in p.items()})
    paths.makedirs()
    return cfg, paths
