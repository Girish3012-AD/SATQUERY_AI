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
        "how far": "distance",
        "distance from": "distance",
        "distance between": "distance",
        "how close": "distance",
        "total area": "area",
        "how much area": "area",
        "covered area": "area",
    }

    # Explicit natural-language visual-question intent.
    # These patterns take precedence over entity keywords such as
    # "building" when the user is asking the VQA model to interpret
    # the image rather than requesting deterministic detection/GIS.
    VQA_PATTERNS = (
        "what is shown",
        "what is visible",
        "what can be seen",
        "describe the image",
        "describe this image",
        "describe what is visible",
        "summarize the image",
        "summarize this image",
        "summarize the building information",
        "using the provided evidence",
        "based on the provided evidence",
        "according to the provided evidence",
    )

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

        explicit_vqa_intent = any(
            pattern in normalized
            for pattern in self.VQA_PATTERNS
        )

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
        #
        # Match spatial intent while distinguishing an actual area
        # measurement from the word "area" used to describe a
        # reference geometry in another operation.
        #
        # Example:
        #   "total building area" -> area
        #   "how far ... from the reference area" -> distance
        # ---------------------------------------------------------
        spatial_matches: list[tuple[int, str, str]] = []

        for keyword, operation in self.SPATIAL_PATTERNS.items():
            if keyword in normalized:
                spatial_matches.append(
                    (len(keyword), keyword, operation)
                )

        spatial_matches.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        has_distance_intent = any(
            operation == "distance"
            for _, _, operation in spatial_matches
        )

        explicit_area_measurement = any(
            phrase in normalized
            for phrase in (
                "total area",
                "how much area",
                "area of",
                "area covered",
                "area covered by",
                "area detected",
                "area occupied",
                "area of the detected",
                "total detected building area",
                "total building area",
                "detected building area",
                "building area",
                "area occupied by buildings",
                "area covered by buildings",
                "area occupied by the detected buildings",
                "area covered by the detected buildings",
            )
        )

        for _, keyword, operation in spatial_matches:
            if (
                operation == "area"
                and has_distance_intent
                and not explicit_area_measurement
            ):
                continue

            if operation not in spatial_operations:
                spatial_operations.append(operation)

        # Explicit area-measurement intent can occur in phrases such as
        # "total detected building area", where the word "area" is
        # meaningful as a GIS measurement but does not match one of the
        # fixed SPATIAL_PATTERNS above.
        #
        # Do not infer GIS area merely because "area" appears in a
        # reference geometry such as "flooded areas".
        if explicit_area_measurement:
            # "building area", "flood area", etc. can describe a
            # reference geometry rather than request an area calculation.
            # If the query already contains a stronger spatial relation,
            # such as distance or intersection, preserve that relation
            # and do not add a second area operation.
            relation_operations = {
                "distance",
                "intersection",
                "buffer",
            }

            if not any(
                operation in relation_operations
                for operation in spatial_operations
            ):
                if "area" not in spatial_operations:
                    spatial_operations.append("area")

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

        if explicit_vqa_intent:
            if "vqa" not in capabilities:
                capabilities.append("vqa")
        elif not any(
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
        elif explicit_vqa_intent:
            task_type = "vqa"
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
