from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class OilSpillDataset(Dataset):
    """
    Dataset for Sentinel-1 SAR images and corresponding oil-spill masks.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_size: int = 512,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Detection dataset: image directory does not exist: "
                f"{self.image_dir}"
            )

        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Detection dataset: mask directory does not exist: "
                f"{self.mask_dir}"
            )

        self.samples = self._build_samples()

        if not self.samples:
            raise RuntimeError(
                "Detection dataset: no matching image/mask pairs were found."
            )

    def _build_samples(self) -> List[Tuple[Path, Path]]:
        samples = []

        image_files = sorted(
            [
                p
                for p in self.image_dir.rglob("*")
                if p.suffix.lower() in {".tif", ".tiff"}
            ]
        )

        for image_path in image_files:
            mask_path = self.mask_dir / image_path.name

            if mask_path.exists():
                samples.append((image_path, mask_path))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        image = self._read_raster(image_path)
        mask = self._read_raster(mask_path)

        return image, mask

    def _read_raster(self, path: Path) -> np.ndarray:
        try:
            with rasterio.open(path) as src:
                array = src.read()

        except Exception as exc:
            raise RuntimeError(
                f"Detection dataset: failed to read {path}"
            ) from exc

        return array
