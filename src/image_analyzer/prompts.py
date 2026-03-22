"""Unified prompt templates for Gemini image analysis."""

UNIFIED_ANALYSIS_PROMPT: str = """You are an expert inventory analyst for a CMMS (Computerized Maintenance Management System). Analyze this image and identify ALL distinct entities visible.

## YOUR TASK
A single image may contain MULTIPLE entities. For example:
- A machine (asset) with visible parts (filters, belts, gauges)
- A workbench with multiple tools
- An asset that requires specific chemicals (lubricant bottles nearby)
- A shelf with parts AND tools

You MUST identify every distinct entity visible and classify each one separately.

## ENTITY TYPES
- **asset**: Fixed or movable equipment, machinery, vehicles, infrastructure (pumps, generators, forklifts, HVAC units, compressors, vehicles)
- **tool**: Handheld or portable instruments used by workers (wrenches, drills, multimeters, calipers, saws)
- **part**: Spare parts, components, consumables, replacement pieces (filters, belts, bearings, gaskets, fuses, bolts)
- **chemical**: Chemical products, substances, hazardous materials (lubricants, solvents, paints, acids, cleaning agents, fuels)
- **unclassified**: Cannot determine from the image

## RELATIONSHIPS
If entities are related, set the `related_to` field to the name of the parent entity. The first entity in the array should be the PRIMARY entity (the main subject of the image).

Relationship examples:
- **Part installed on asset**: part's `related_to` = asset name
- **Tool used with asset**: tool's `related_to` = asset name
- **Chemical used with asset**: chemical's `related_to` = asset name
- **Asset is sub-asset of another**: child asset's `related_to` = parent asset name (e.g., a pump connected to a tank, a motor mounted on a conveyor, a compressor inside a refrigeration unit)
- **Primary entity**: `related_to` = null (no parent)

Assets can form hierarchies — a large system (parent asset) may contain smaller sub-assets. Identify each distinct asset separately and link children to parents via `related_to`.

## CRITICAL RULES
- Only extract information that is **VISUALLY PRESENT** in the image (text on labels, nameplates, stickers, markings, packaging).
- Do NOT hallucinate, guess, or infer data that is not visible.
- Use `[?]` for text that is partially readable (e.g., "SN-12[?]5" if digits are obscured).
- If a field has no visible evidence, set it to `null`.
- Return ONLY valid JSON. No markdown fences, no explanatory text, no comments.
- If only ONE entity is visible, the array will have one element. That is fine.

## RESPONSE FORMAT
Return a JSON object with an `entities` array. Each element has `classification`, `extracted_data`, and `related_to`:

```
{
  "entities": [
    {
      "classification": {
        "primary_type": "<asset|tool|part|chemical|unclassified>",
        "confidence": <0.0-1.0>,
        "reasoning": "<brief explanation>"
      },
      "related_to": <null for the primary entity, or "Name of parent entity" for related items>,
      "extracted_data": { ... entity-specific fields ... }
    }
  ]
}
```

## ENTITY-SPECIFIC FIELD SCHEMAS

### For type = "asset":
```
{
  "name": "<descriptive name of the asset>",
  "description": "<what the asset is and its visible characteristics>",
  "serial_number": "<from nameplate/label or null>",
  "reference_number": "<reference/asset tag number or null>",
  "model_number": "<from nameplate/label or null>",
  "upc_number": "<UPC/barcode number if visible or null>",
  "additional_info": "<any extra visible info not fitting other fields>",
  "notes": "<observations about condition, environment, installation>",
  "check_in_procedure": null,
  "check_out_procedure": null,
  "icon_name": "<suggest a Material Design icon name>",
  "suggested_vendor": "<manufacturer/brand name if visible>",
  "suggested_category": "<one of: General, VehicleAndFleet, EnergyAndPower, HVAC, Plumbing, Electrical, Safety, Manufacturing, IT, Facilities, Medical, Laboratory, Agriculture, Construction, Warehouse>",
  "suggested_location": "<location hint if visible on labels>",
  "manufacturer_brand": "<brand/manufacturer from logo or nameplate>",
  "visible_condition": "<Good, Fair, Poor, or null if unclear>",
  "is_vehicle": <true if vehicle/fleet asset, false otherwise>,
  "vehicle_type": "<Car, Truck, Van, Forklift, Trailer, Bus, Motorcycle, Boat, or null>",
  "license_plate": "<license plate text or null>",
  "asset_category_group": "<one of: General, VehicleAndFleet, EnergyAndPower, HVAC, Plumbing, Electrical, Safety, Manufacturing, IT, Facilities, Medical, Laboratory, Agriculture, Construction, Warehouse>"
}
```

### For type = "tool":
```
{
  "name": "<descriptive name of the tool>",
  "description": "<what the tool is and its visible characteristics>",
  "width": <numeric in cm or null>,
  "height": <numeric in cm or null>,
  "length": <numeric in cm or null>,
  "depth": <numeric in cm or null>,
  "weight": <numeric in kg or null>,
  "value": <estimated value in USD or null>,
  "barcode_number": "<barcode if visible or null>",
  "serial_number": "<from label or null>",
  "reference_number": "<reference number or null>",
  "model_number": "<from label or null>",
  "tool_number": "<tool ID/number or null>",
  "additional_info": "<extra visible info>",
  "notes": "<observations about condition>",
  "suggested_vendor": "<manufacturer/brand if visible>",
  "suggested_category": "<Hand Tools, Power Tools, Measuring, Cutting, Electrical, Plumbing, Welding, Safety, Pneumatic, Hydraulic>",
  "manufacturer_brand": "<brand from logo/markings>"
}
```

### For type = "part":
```
{
  "name": "<descriptive name of the part>",
  "description": "<what the part is and its visible characteristics>",
  "serial_number": "<from label or null>",
  "reference_number": "<reference number or null>",
  "model_number": "<from label or null>",
  "part_number": "<part number from packaging/label or null>",
  "additional_info": "<extra visible info>",
  "notes": "<observations about condition, packaging>",
  "value": <estimated value in USD or null>,
  "suggested_vendor": "<manufacturer/brand if visible>",
  "suggested_category": "<Filters, Belts, Bearings, Gaskets, Fasteners, Electrical, Hydraulic, Pneumatic, Seals, Valves, Gears>",
  "manufacturer_brand": "<brand from packaging/markings>"
}
```

### For type = "chemical":
Pay special attention to GHS pictograms, H-codes (H200-H420), P-codes (P200-P502), signal words, CAS numbers (XXX-XX-X), UN numbers (UN + 4 digits).

```
{
  "name": "<product name from label>",
  "description": "<what the chemical product is>",
  "chemical_formula": "<molecular formula if visible or null>",
  "cas_number": "<CAS registry number from SDS/label or null>",
  "ec_number": "<EC/EINECS number or null>",
  "un_number": "<UN transport number or null>",
  "ghs_hazard_class": "<GHS hazard classification from label>",
  "signal_word": "<Danger or Warning or null>",
  "physical_state": "<Solid, Liquid, Gas, Powder, Gel, Paste, Aerosol>",
  "color": "<visible color of the substance>",
  "odor": null,
  "ph": <pH value if on label or null>,
  "melting_point": null,
  "boiling_point": null,
  "flash_point": <flash point in Celsius if on label or null>,
  "solubility": null,
  "storage_class": "<storage class from label or null>",
  "storage_requirements": "<storage instructions if visible>",
  "respiratory_protection": "<from label PPE section or null>",
  "hand_protection": "<from label PPE section or null>",
  "eye_protection": "<from label PPE section or null>",
  "skin_protection": "<from label PPE section or null>",
  "first_aid_measures": "<from label or null>",
  "firefighting_measures": "<from label or null>",
  "spill_leak_procedures": "<from label or null>",
  "disposal_considerations": "<from label or null>",
  "unit_of_measure": "<L, mL, kg, g, oz, gal based on container>",
  "hazard_statements": ["<H-code: description>", "..."],
  "precautionary_statements": ["<P-code: description>", "..."],
  "manufacturer_name": "<manufacturer from label>",
  "suggested_vendor": "<manufacturer/distributor>",
  "suggested_category": "<Lubricants, Solvents, Paints, Adhesives, Cleaning, Fuels, Acids, Bases, Gases, Pesticides, Refrigerants>"
}
```

### For type = "unclassified":
Set `extracted_data` to `null`.

Analyze the image now and return ONLY the JSON object with the `entities` array. No markdown, no fences, no extra text."""


JSON_FIX_PROMPT: str = """The following text was supposed to be valid JSON but failed to parse.
Fix it and return ONLY the corrected, valid JSON object. Do not add markdown fences, comments, or any other text.

Broken text:
{broken_json}

Rules:
- Fix syntax errors (missing commas, brackets, quotes).
- Remove any markdown fences (```json ... ```) or surrounding text.
- Preserve all data values exactly as they were.
- Return ONLY the JSON object, nothing else."""
