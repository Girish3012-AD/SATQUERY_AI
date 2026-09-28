import re

with open('src/executor/gis_executor.py', 'r', encoding='utf-8') as f:
    code = f.read()

replacement = '''
        if operation == "buffer":
            distance_m = parameters.get("distance_m")

            if distance_m is None:
                raise ValueError(
                    "Buffer operation requires 'distance_m'."
                )

            source = geometry_evidence[-1]
            distance_m = self._validate_distance_parameter(distance_m)
            
            orig_crs = self._crs_from_evidence(source)
            if orig_crs is None:
                raise ValueError(
                    f"GIS {operation} requires CRS metadata on evidence '{source.evidence_id}'."
                )

            geometry = self._geometry_from_evidence(source)
            proj_crs = self._get_projected_crs(geometry, orig_crs)
            if proj_crs is None:
                self._validate_projected_crs(source, operation)

            proj_geom = self._reproject(geometry, orig_crs, proj_crs)
            result_proj_geom = buffer_geometry(proj_geom, distance_m)
            result_geometry = self._reproject(result_proj_geom, proj_crs, orig_crs)

            return self._make_evidence(
                operation=operation,
                source_evidence=[source],
                result_geometry=result_geometry,
                measurement={
                    "operation": "buffer",
                    "distance_m": distance_m,
                    "area_m2": calculate_area(result_proj_geom),
                    "crs": orig_crs.to_string(),
                    "projected_crs": proj_crs.to_string() if orig_crs != proj_crs else None
                },
            )

        if operation == "intersection":
            if len(geometry_evidence) < 2:
                raise ValueError(
                    "Intersection requires at least two geometry-bearing evidence objects."
                )

            first_evidence = geometry_evidence[-2]
            second_evidence = geometry_evidence[-1]
            
            crs1 = self._crs_from_evidence(first_evidence)
            crs2 = self._crs_from_evidence(second_evidence)
            if crs1 is None or crs2 is None:
                raise ValueError(f"GIS {operation} requires CRS metadata on evidence.")

            first = self._geometry_from_evidence(first_evidence)
            second = self._geometry_from_evidence(second_evidence)

            proj_crs1 = self._get_projected_crs(first, crs1)
            proj_crs2 = self._get_projected_crs(second, crs2)
            
            if proj_crs1 is None or proj_crs2 is None or (proj_crs1 != proj_crs2 and crs1.is_projected and crs2.is_projected):
                self._validate_compatible_crs(first_evidence, second_evidence, operation)
            
            common_proj_crs = proj_crs1 if proj_crs1 else crs1

            proj_first = self._reproject(first, crs1, common_proj_crs)
            proj_second = self._reproject(second, crs2, common_proj_crs)

            result_proj_geom = intersect_geometries(proj_first, proj_second)
            result_geometry = self._reproject(result_proj_geom, common_proj_crs, crs1)

            return self._make_evidence(
                operation=operation,
                source_evidence=geometry_evidence[-2:],
                result_geometry=result_geometry,
                measurement={
                    "operation": "intersection",
                    "area_m2": calculate_area(result_proj_geom),
                    "is_empty": bool(result_geometry.is_empty),
                    "crs": crs1.to_string(),
                    "projected_crs": common_proj_crs.to_string() if crs1 != common_proj_crs else None
                },
            )

        if operation == "distance":
            if len(geometry_evidence) < 2:
                raise ValueError("Distance requires at least two geometry-bearing evidence objects.")

            first_evidence = geometry_evidence[-2]
            second_evidence = geometry_evidence[-1]
            
            crs1 = self._crs_from_evidence(first_evidence)
            crs2 = self._crs_from_evidence(second_evidence)
            if crs1 is None or crs2 is None:
                raise ValueError(f"GIS {operation} requires CRS metadata on evidence.")

            first = self._geometry_from_evidence(first_evidence)
            second = self._geometry_from_evidence(second_evidence)

            proj_crs1 = self._get_projected_crs(first, crs1)
            proj_crs2 = self._get_projected_crs(second, crs2)
            if proj_crs1 is None or proj_crs2 is None or (proj_crs1 != proj_crs2 and crs1.is_projected and crs2.is_projected):
                self._validate_compatible_crs(first_evidence, second_evidence, operation)
                
            common_proj_crs = proj_crs1 if proj_crs1 else crs1
            
            proj_first = self._reproject(first, crs1, common_proj_crs)
            proj_second = self._reproject(second, crs2, common_proj_crs)
            distance = calculate_distance(proj_first, proj_second)

            return self._make_evidence(
                operation=operation,
                source_evidence=geometry_evidence[-2:],
                result_geometry=None,
                measurement={
                    "operation": "distance",
                    "distance_m": float(distance),
                    "crs": crs1.to_string(),
                    "projected_crs": common_proj_crs.to_string() if crs1 != common_proj_crs else None
                },
                result={"distance_m": float(distance)},
            )

        if operation == "area":
            source = geometry_evidence[-1]
            orig_crs = self._crs_from_evidence(source)
            if orig_crs is None:
                raise ValueError(f"GIS {operation} requires CRS metadata.")

            geometry = self._geometry_from_evidence(source)
            proj_crs = self._get_projected_crs(geometry, orig_crs)
            if proj_crs is None:
                self._validate_projected_crs(source, operation)

            proj_geom = self._reproject(geometry, orig_crs, proj_crs)
            area = calculate_area(proj_geom)

            return self._make_evidence(
                operation=operation,
                source_evidence=[source],
                result_geometry=geometry,
                measurement={
                    "operation": "area",
                    "area_m2": float(area),
                    "crs": orig_crs.to_string(),
                    "projected_crs": proj_crs.to_string() if orig_crs != proj_crs else None
                },
                result={"area_m2": float(area)},
            )
'''

pattern = re.compile(r'        if operation == "buffer":.*?raise ValueError\(\s*"Unsupported GIS operation: {operation}"\s*\)', re.DOTALL)
new_code, count = pattern.subn(replacement.strip() + '\n\n        raise ValueError(\n            f"Unsupported GIS operation: {operation}"\n        )', code)

if count > 0:
    with open('src/executor/gis_executor.py', 'w', encoding='utf-8') as f:
        f.write(new_code)
    print("Successfully replaced.")
else:
    print("Could not find the pattern to replace.")
