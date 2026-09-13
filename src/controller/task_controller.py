import re

from src.schemas import TaskSpec


class TaskController:
    """
    Deterministic baseline controller for converting a natural-language
    remote-sensing query into a structured SATQuery TaskSpec.

    This controller is intentionally model-independent. The future
    Qwen controller must preserve the same TaskSpec contract.
    """

    TEMPORAL_KEYWORDS = {
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

    SAR_KEYWORDS = {
        "sar",
        "radar",
        "risat",
        "synthetic aperture radar",
    }

    OPTICAL_KEYWORDS = {
        "optical",
        "multispectral",
        "sentinel-2",
        "landsat",
        "cartosat",
    }

    ENTITY_CAPABILITIES = {
        "building": "building_detection",
        "buildings": "building_detection",
        "house": "building_detection",
        "houses": "building_detection",
        "structure": "building_detection",
        "structures": "building_detection",
        "flood": "flood_detection",
        "flooded": "flood_detection",
        "flooding": "flood_detection",
        "water": "water_detection",
        "vegetation": "vegetation_detection",
        "crop": "crop_detection",
        "crops": "crop_detection",
        "road": "road_detection",
        "roads": "road_detection",
    }

    SPATIAL_PATTERNS = {
        "within": "buffer",
        "inside": "buffer",
        "near": "buffer",
        "buffer": "buffer",
        "intersect": "intersection",
        "intersection": "intersection",
        "overlap": "intersection",
        "distance": "distance",
    }

    def build_task_spec(
        self,
        query: str,
        input_count: int,
    ) -> TaskSpec:
        """Build a structured TaskSpec from a natural-language query."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if input_count < 0:
            raise ValueError("input_count cannot be negative.")

        normalized = query.lower().strip()

        capabilities: list[str] = []
        modalities: list[str] = []
        spatial_operations: list[str] = []
        parameters: dict[str, str | int | float | bool] = {}

        # ---------------------------------------------------------
        # Temporal analysis
        # ---------------------------------------------------------
        requires_temporal_pair = any(
            keyword in normalized
            for keyword in self.TEMPORAL_KEYWORDS
        )

        if requires_temporal_pair:
            capabilities.append("temporal_analysis")

        # ---------------------------------------------------------
        # Sensor / modality requirements
        # ---------------------------------------------------------
        if any(
            keyword in normalized
            for keyword in self.SAR_KEYWORDS
        ):
            modalities.append("sar")
            capabilities.append("sar_analysis")

        if any(
            keyword in normalized
            for keyword in self.OPTICAL_KEYWORDS
        ):
            modalities.append("optical")

        # ---------------------------------------------------------
        # Detect semantic entities
        # ---------------------------------------------------------
        for keyword, capability in self.ENTITY_CAPABILITIES.items():
            if keyword in normalized:
                if capability not in capabilities:
                    capabilities.append(capability)

        # ---------------------------------------------------------
        # Spatial operations
        # ---------------------------------------------------------
        for keyword, operation in self.SPATIAL_PATTERNS.items():
            if keyword in normalized:
                if operation not in spatial_operations:
                    spatial_operations.append(operation)

        # ---------------------------------------------------------
        # Extract distance
        #
        # Examples:
        #   500 meters
        #   500 m
        #   2 km
        #   1.5 kilometers
        # ---------------------------------------------------------
        distance_match = re.search(
            r"(\d+(?:\.\d+)?)\s*"
            r"(meters?|metres?|m|km|kilometers?|kilometres?)",
            normalized,
        )

        if distance_match:
            value = float(distance_match.group(1))
            unit = distance_match.group(2)

            if (
                unit.startswith("km")
                or unit.startswith("kilometer")
                or unit.startswith("kilometre")
            ):
                value *= 1000

            parameters["distance_m"] = value

        # ---------------------------------------------------------
        # Generic visual interpretation
        #
        # Only add VQA when the query does not already describe a
        # more specific specialist capability.
        # ---------------------------------------------------------
        specialist_capabilities = {
            "temporal_analysis",
            "sar_analysis",
            "building_detection",
            "flood_detection",
            "water_detection",
            "vegetation_detection",
            "crop_detection",
            "road_detection",
        }

        if not any(
            capability in specialist_capabilities
            for capability in capabilities
        ):
            capabilities.append("vqa")

        # ---------------------------------------------------------
        # Determine broad task type
        # ---------------------------------------------------------
        specialist_present = any(
            capability in specialist_capabilities
            for capability in capabilities
        )

        if requires_temporal_pair:
            task_type = "temporal_analysis"
        elif spatial_operations:
            task_type = "spatial_analysis"
        elif specialist_present:
            task_type = "specialized_analysis"
        else:
            task_type = "vqa"

        # ---------------------------------------------------------
        # Remove duplicates while preserving order
        # ---------------------------------------------------------
        capabilities = list(dict.fromkeys(capabilities))
        modalities = list(dict.fromkeys(modalities))

        return TaskSpec(
            task_id="TASK-001",
            query=query,
            task_type=task_type,
            required_capabilities=capabilities,
            required_modalities=modalities,
            input_count=input_count,
            requires_temporal_pair=requires_temporal_pair,
            spatial_operations=spatial_operations,
            parameters=parameters,
        )
