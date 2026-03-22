"""Unified prompt templates for Gemini image analysis."""

UNIFIED_ANALYSIS_PROMPT: str = """You are an expert inventory analyst. Analyze this image and perform TWO tasks:

## TASK 1: CLASSIFY the image
Determine which category this image belongs to:
- **asset**: Fixed or movable equipment, machinery, vehicles, infrastructure (pumps, generators, forklifts, HVAC units, compressors, vehicles)
- **tool**: Handheld or portable instruments used by workers (wrenches, drills, multimeters, calipers, saws)
- **part**: Spare parts, components, consumables, replacement pieces (filters, belts, bearings, gaskets, fuses, bolts)
- **chemical**: Chemical products, substances, hazardous materials (lubricants, solvents, paints, acids, cleaning agents, fuels)
- **unclassified**: Cannot determine from the image

Provide a confidence score from 0.0 to 1.0 and brief reasoning.

## TASK 2: EXTRACT entity-specific fields
Based on your classification, extract ALL visually present information into the matching JSON schema below.

### CRITICAL RULES:
- Only extract information that is **VISUALLY PRESENT** in the image (text on labels, nameplates, stickers, markings, packaging).
- Do NOT hallucinate, guess, or infer data that is not visible.
- Use `[?]` for text that is partially readable (e.g., "SN-12[?]5" if digits are obscured).
- If a field has no visible evidence, set it to `null`.
- Return ONLY valid JSON. No markdown fences, no explanatory text, no comments.

### JSON Schemas by classification:

#### If classification = "asset":
```
{
  "classification": {
    "primary_type": "asset",
    "confidence": <0.0-1.0>,
    "secondary_type": <null or "tool"|"part">,
    "reasoning": "<brief explanation>"
  },
  "extracted_data": {
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
}
```

#### If classification = "tool":
```
{
  "classification": {
    "primary_type": "tool",
    "confidence": <0.0-1.0>,
    "secondary_type": <null or "asset"|"part">,
    "reasoning": "<brief explanation>"
  },
  "extracted_data": {
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
}
```

#### If classification = "part":
```
{
  "classification": {
    "primary_type": "part",
    "confidence": <0.0-1.0>,
    "secondary_type": <null or "tool"|"asset">,
    "reasoning": "<brief explanation>"
  },
  "extracted_data": {
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
}
```

#### If classification = "chemical":
For chemicals, pay special attention to:
- **GHS pictograms**: Flame, Exploding Bomb, Oxidizer, Gas Cylinder, Corrosion, Skull & Crossbones, Exclamation Mark, Health Hazard, Environment
- **H-codes** (Hazard statements): H200-H420 range (e.g., H225 = Highly flammable liquid, H302 = Harmful if swallowed, H314 = Severe skin burns)
- **P-codes** (Precautionary statements): P200-P502 range (e.g., P210 = Keep away from heat, P280 = Wear protective gloves, P305+P351+P338 = Eye wash instructions)
- **Signal words**: "Danger" or "Warning"
- **CAS numbers**: Format XXX-XX-X or XXXX-XX-X
- **UN numbers**: Format UN followed by 4 digits (e.g., UN1203)

```
{
  "classification": {
    "primary_type": "chemical",
    "confidence": <0.0-1.0>,
    "secondary_type": null,
    "reasoning": "<brief explanation>"
  },
  "extracted_data": {
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
}
```

#### If classification = "unclassified":
```
{
  "classification": {
    "primary_type": "unclassified",
    "confidence": <0.0-1.0>,
    "secondary_type": null,
    "reasoning": "<explain why the image could not be classified>"
  },
  "extracted_data": null
}
```

Analyze the image now and return ONLY the JSON object. No markdown, no fences, no extra text."""


JSON_FIX_PROMPT: str = """The following text was supposed to be valid JSON but failed to parse.
Fix it and return ONLY the corrected, valid JSON object. Do not add markdown fences, comments, or any other text.

Broken text:
{broken_json}

Rules:
- Fix syntax errors (missing commas, brackets, quotes).
- Remove any markdown fences (```json ... ```) or surrounding text.
- Preserve all data values exactly as they were.
- Return ONLY the JSON object, nothing else."""
