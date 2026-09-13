from dataclasses import dataclass

from src.schemas.image import ImageRef


@dataclass
class CompatibilityResult:
    """Result of checking whether an input supports a query."""

    supported: bool
    reason: str
    required_modalities: list[str]
    available_modalities: list[str]
    requires_temporal_pair: bool = False


class CompatibilityChecker:
    """Check whether available imagery can support a SATQuery task."""

    def check(
        self,
        query: str,
        images: list[ImageRef],
    ) -> CompatibilityResult:
        """
        Determine whether the supplied imagery is compatible
        with the user's natural-language query.
        """

        if not images:
            return CompatibilityResult(
                supported=False,
                reason="No input images were provided.",
                required_modalities=[],
                available_modalities=[],
            )

        query_lower = query.lower()

        available_modalities = sorted(
            {
                image.modality.lower()
                for image in images
                if image.modality
            }
        )

        temporal_keywords = {
            "change",
            "changed",
            "change detection",
            "before and after",
            "newly constructed",
            "construction",
            "growth",
            "loss",
            "increase",
            "decrease",
        }

        sar_keywords = {
            "sar",
            "radar",
            "risat",
            "synthetic aperture radar",
        }

        optical_keywords = {
            "optical",
            "multispectral",
            "multispectral image",
            "sentinel-2",
            "landsat",
            "cartosat",
        }

        requires_temporal_pair = any(
            keyword in query_lower
            for keyword in temporal_keywords
        )

        requires_sar = any(
            keyword in query_lower
            for keyword in sar_keywords
        )

        requires_optical = any(
            keyword in query_lower
            for keyword in optical_keywords
        )

        required_modalities = []

        if requires_sar:
            required_modalities.append("sar")

        if requires_optical:
            required_modalities.append("optical")

        if requires_temporal_pair and len(images) < 2:
            return CompatibilityResult(
                supported=False,
                reason=(
                    "This query requires temporal comparison, "
                    "but fewer than two images were provided."
                ),
                required_modalities=required_modalities,
                available_modalities=available_modalities,
                requires_temporal_pair=True,
            )

        if requires_sar and "sar" not in available_modalities:
            return CompatibilityResult(
                supported=False,
                reason=(
                    "This query requires SAR imagery, "
                    "but no SAR image was provided."
                ),
                required_modalities=required_modalities,
                available_modalities=available_modalities,
                requires_temporal_pair=requires_temporal_pair,
            )

        if requires_optical and "optical" not in available_modalities:
            return CompatibilityResult(
                supported=False,
                reason=(
                    "This query requires optical imagery, "
                    "but no optical image was provided."
                ),
                required_modalities=required_modalities,
                available_modalities=available_modalities,
                requires_temporal_pair=requires_temporal_pair,
            )

        return CompatibilityResult(
            supported=True,
            reason="Input imagery is compatible with the query.",
            required_modalities=required_modalities,
            available_modalities=available_modalities,
            requires_temporal_pair=requires_temporal_pair,
        )
