# Demo Screenshot Manifest

| Filename | Dimensions | File Size | Workflow Used | Generation Mechanism |
|----------|------------|-----------|---------------|----------------------|
| `final_validation/01_single_vqa.png` | 1440x1080 | 613 KB | Single-image VQA | Playwright Automated |
| `final_validation/02_water_grounding.png` | 1440x1080 | 684 KB | Water Grounding | Playwright Automated |
| `final_validation/03_temporal_change.png` | 1440x1080 | 644 KB | Bi-temporal Change | Playwright Automated |
| `final_validation/04_optical_sar.png` | 1440x1080 | 752 KB | Optical + SAR | Playwright Automated |
| `final_validation/05_hero.png` | 1440x1080 | 337 KB | Hero Geographic Reasoning | Playwright Automated |

All screenshots were generated directly from the live `http://localhost:8000` SATQuery application running the full backend orchestrator natively. No fabricated traces or mock UI states were used.

### Hero Geographic Reasoning (`05_hero.png`) details
**Workflow:**
Natural-language geographic query
→ Query Planner
→ input binding
→ water detection
→ temporal/change analysis
→ building detection
→ CRS-safe 500m buffer
→ spatial intersection
→ evidence generation

**State:**
LIVE SUCCESSFULLY VERIFIED
