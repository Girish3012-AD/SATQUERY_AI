from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import planetary_computer
import requests

from .sentinel1_stac import (
    Sentinel1STACDiscovery,
    Sentinel1Candidate,
)


DEFAULT_OUTPUT_DIR = Path(
    "data/remote_sensing/sentinel1/raw"
)

DEFAULT_METADATA_DIR = Path(
    "data/remote_sensing/sentinel1/metadata"
)


class Sentinel1Acquisition:
    """
    Acquire a real Sentinel-1 VV asset discovered through STAC.

    Features:
    - Planetary Computer signed assets
    - resumable downloads
    - temporary-file protection
    - download progress reporting
    - provenance metadata

    This component does not claim RISAT validation.
    """

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        metadata_dir: Path = DEFAULT_METADATA_DIR,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.metadata_dir = Path(metadata_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_filename(item_id: str) -> str:
        return "".join(
            character
            if character.isalnum() or character in "._-"
            else "_"
            for character in item_id
        )

    def download_vv(
        self,
        candidate: Sentinel1Candidate,
        timeout: int = 120,
    ) -> Path:
        if "VV" not in candidate.assets:
            raise ValueError(
                f"Candidate {candidate.item_id} does not contain a VV asset."
            )

        unsigned_url = candidate.assets["VV"]
        signed_url = planetary_computer.sign(unsigned_url)

        filename = (
            f"{self._safe_filename(candidate.item_id)}_VV.tif"
        )

        output_path = self.output_dir / filename
        temporary_path = output_path.with_suffix(".download")

        existing_size = (
            temporary_path.stat().st_size
            if temporary_path.exists()
            else 0
        )

        print("============================================================")
        print("SENTINEL-1 VV DOWNLOAD")
        print("============================================================")
        print("Item:", candidate.item_id)
        print("Output:", output_path)
        print("Existing partial bytes:", existing_size)

        headers: dict[str, str] = {}

        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            print(
                f"Resuming from byte {existing_size:,}"
            )
        else:
            print("Starting new download")

        start_time = time.perf_counter()

        with requests.get(
            signed_url,
            headers=headers,
            stream=True,
            timeout=timeout,
        ) as response:

            response.raise_for_status()

            status = response.status_code
            content_length = response.headers.get(
                "Content-Length"
            )

            print("HTTP status:", status)
            print("Response size:", content_length)
            print(
                "Content-Range:",
                response.headers.get("Content-Range"),
            )

            # If we requested a resume but the server returned 200,
            # it ignored the Range header. Restart safely rather than
            # corrupting the raster by appending from byte zero.
            if existing_size > 0 and status == 200:
                print(
                    "Server did not honor Range request."
                )
                print(
                    "Restarting download from byte zero."
                )

                temporary_path.unlink(missing_ok=True)

                response.close()

                with requests.get(
                    signed_url,
                    stream=True,
                    timeout=timeout,
                ) as restart_response:
                    restart_response.raise_for_status()

                    total_size = int(
                        restart_response.headers.get(
                            "Content-Length",
                            "0",
                        )
                    )

                    downloaded = 0

                    with temporary_path.open("wb") as destination:
                        for chunk in restart_response.iter_content(
                            chunk_size=4 * 1024 * 1024
                        ):
                            if not chunk:
                                continue

                            destination.write(chunk)
                            downloaded += len(chunk)

                            if downloaded % (
                                32 * 1024 * 1024
                            ) < len(chunk):
                                print(
                                    f"Downloaded "
                                    f"{downloaded:,}"
                                    f"/{total_size:,} bytes"
                                )

            else:
                total_size = None

                if content_length is not None:
                    content_length_int = int(content_length)

                    if status == 206:
                        total_size = existing_size + content_length_int
                    else:
                        total_size = content_length_int

                downloaded = existing_size

                file_mode = "ab" if status == 206 else "wb"

                with temporary_path.open(
                    file_mode
                ) as destination:

                    for chunk in response.iter_content(
                        chunk_size=4 * 1024 * 1024
                    ):
                        if not chunk:
                            continue

                        destination.write(chunk)
                        downloaded += len(chunk)

                        if downloaded % (
                            32 * 1024 * 1024
                        ) < len(chunk):
                            if total_size:
                                percentage = (
                                    downloaded
                                    / total_size
                                    * 100
                                )
                                print(
                                    f"Downloaded "
                                    f"{downloaded:,}/"
                                    f"{total_size:,} bytes "
                                    f"({percentage:.1f}%)"
                                )
                            else:
                                print(
                                    f"Downloaded "
                                    f"{downloaded:,} bytes"
                                )

        if not temporary_path.exists():
            raise RuntimeError(
                "Download completed without creating "
                "the temporary file."
            )

        final_size = temporary_path.stat().st_size

        if final_size == 0:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Downloaded Sentinel-1 asset is empty."
            )

        elapsed = time.perf_counter() - start_time

        print()
        print("Temporary file size:", final_size, "bytes")
        print("Download elapsed:", f"{elapsed:.2f}", "seconds")

        if total_size is not None and final_size != total_size:
            raise RuntimeError(
                "Downloaded file size does not match the "
                f"expected size. expected={total_size}, "
                f"actual={final_size}"
            )

        shutil.move(
            str(temporary_path),
            str(output_path),
        )

        metadata: dict[str, Any] = {
            "mission": "Sentinel-1",
            "sensor": "Sentinel-1 SAR",
            "modality": "sar",
            "collection": candidate.collection,
            "item_id": candidate.item_id,
            "datetime": candidate.datetime,
            "bbox": candidate.bbox,
            "selected_polarization": "VV",
            "asset_url": unsigned_url,
            "local_path": str(output_path.resolve()),
            "file_size_bytes": final_size,
            "data_status": "real_validation",
            "risat_validated": False,
            "access_method": (
                "planetary_computer_signed_asset"
            ),
            "download_elapsed_seconds": elapsed,
            "warning": (
                "This is real Sentinel-1 SAR data. "
                "It does not constitute RISAT validation."
            ),
        }

        metadata_path = (
            self.metadata_dir
            / f"{self._safe_filename(candidate.item_id)}_VV.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
            )

        print()
        print("============================================================")
        print("DOWNLOAD COMPLETE")
        print("============================================================")
        print("File:", output_path)
        print("Size:", final_size, "bytes")
        print("Metadata:", metadata_path)

        return output_path

    def acquire_best(
        self,
        bbox: list[float],
        datetime_range: str,
    ) -> Path:
        discovery = Sentinel1STACDiscovery()

        candidate = discovery.search_best(
            bbox=bbox,
            datetime_range=datetime_range,
        )

        print()
        print("============================================================")
        print("SELECTED SENTINEL-1 SCENE")
        print("============================================================")
        print("Item ID:", candidate.item_id)
        print("Datetime:", candidate.datetime)
        print("Assets:", sorted(candidate.assets.keys()))

        return self.download_vv(candidate)
