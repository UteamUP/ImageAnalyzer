"""Pydantic models mirroring backend GenerateInventoryFieldsModels.cs DTOs."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    ASSET = "asset"
    TOOL = "tool"
    PART = "part"
    CHEMICAL = "chemical"
    UNCLASSIFIED = "unclassified"


class ClassificationResult(BaseModel):
    primary_type: EntityType
    confidence: float = Field(ge=0.0, le=1.0)
    secondary_type: Optional[EntityType] = None
    reasoning: str = ""


# --- Entity-specific extracted data (mirrors GenerateInventoryFieldsModels.cs) ---


class ExtractedAssetData(BaseModel):
    """Mirrors GeneratedAssetData from backend."""

    name: str
    description: str = ""
    serial_number: Optional[str] = None
    reference_number: Optional[str] = None
    model_number: Optional[str] = None
    upc_number: Optional[str] = None
    additional_info: Optional[str] = None
    notes: Optional[str] = None
    check_in_procedure: Optional[str] = None
    check_out_procedure: Optional[str] = None
    icon_name: Optional[str] = None
    suggested_vendor: Optional[str] = None
    suggested_category: Optional[str] = None
    suggested_location: Optional[str] = None
    # Extended fields for image analysis
    manufacturer_brand: Optional[str] = None
    visible_condition: Optional[str] = None
    is_vehicle: bool = False
    vehicle_type: Optional[str] = None
    license_plate: Optional[str] = None


class ExtractedToolData(BaseModel):
    """Mirrors GeneratedToolData from backend."""

    name: str
    description: Optional[str] = None
    width: Optional[float] = None
    height: Optional[float] = None
    length: Optional[float] = None
    depth: Optional[float] = None
    weight: Optional[float] = None
    value: Optional[float] = None
    barcode_number: Optional[str] = None
    serial_number: Optional[str] = None
    reference_number: Optional[str] = None
    model_number: Optional[str] = None
    tool_number: Optional[str] = None
    additional_info: Optional[str] = None
    notes: Optional[str] = None
    suggested_vendor: Optional[str] = None
    suggested_category: Optional[str] = None
    # Extended
    manufacturer_brand: Optional[str] = None


class ExtractedPartData(BaseModel):
    """Mirrors GeneratedPartData from backend."""

    name: str
    description: Optional[str] = None
    serial_number: Optional[str] = None
    reference_number: Optional[str] = None
    model_number: Optional[str] = None
    part_number: Optional[str] = None
    additional_info: Optional[str] = None
    notes: Optional[str] = None
    value: Optional[float] = None
    suggested_vendor: Optional[str] = None
    suggested_category: Optional[str] = None
    # Extended
    manufacturer_brand: Optional[str] = None


class ExtractedChemicalData(BaseModel):
    """Mirrors GeneratedChemicalData from backend."""

    name: str
    description: str = ""
    chemical_formula: Optional[str] = None
    cas_number: Optional[str] = None
    ec_number: Optional[str] = None
    un_number: Optional[str] = None
    ghs_hazard_class: Optional[str] = None
    signal_word: Optional[str] = None
    physical_state: Optional[str] = None
    color: Optional[str] = None
    odor: Optional[str] = None
    ph: Optional[float] = None
    melting_point: Optional[float] = None
    boiling_point: Optional[float] = None
    flash_point: Optional[float] = None
    solubility: Optional[str] = None
    storage_class: Optional[str] = None
    storage_requirements: Optional[str] = None
    respiratory_protection: Optional[str] = None
    hand_protection: Optional[str] = None
    eye_protection: Optional[str] = None
    skin_protection: Optional[str] = None
    first_aid_measures: Optional[str] = None
    firefighting_measures: Optional[str] = None
    spill_leak_procedures: Optional[str] = None
    disposal_considerations: Optional[str] = None
    unit_of_measure: str = "L"
    hazard_statements: list[str] = Field(default_factory=list)
    precautionary_statements: list[str] = Field(default_factory=list)
    manufacturer_name: Optional[str] = None
    suggested_vendor: Optional[str] = None
    suggested_category: Optional[str] = None


# --- Composite result ---


class ImageAnalysisResult(BaseModel):
    """Complete analysis result for a single image."""

    image_path: str
    original_filename: str
    file_hash_sha256: str
    perceptual_hash: str = ""
    classification: ClassificationResult
    extracted_data: Optional[
        ExtractedAssetData | ExtractedToolData | ExtractedPartData | ExtractedChemicalData
    ] = None
    exif_metadata: dict = Field(default_factory=dict)
    flagged_for_review: bool = False
    review_reason: Optional[str] = None
    processed_at: datetime = Field(default_factory=datetime.now)
    # iPhone edit pairing
    paired_images: list[str] = Field(default_factory=list)


class ImageGroup(BaseModel):
    """A group of images depicting the same physical item."""

    primary: ImageAnalysisResult
    members: list[ImageAnalysisResult] = Field(default_factory=list)
    group_confidence: float = 0.0

    @property
    def all_image_paths(self) -> list[str]:
        paths = [self.primary.image_path]
        for m in self.members:
            paths.append(m.image_path)
            paths.extend(m.paired_images)
        paths.extend(self.primary.paired_images)
        return paths

    @property
    def all_original_filenames(self) -> list[str]:
        names = [self.primary.original_filename]
        for m in self.members:
            names.append(m.original_filename)
        return names


# --- CSV column definitions matching backend DTOs ---

ASSET_CSV_COLUMNS = [
    "name", "description", "serial_number", "reference_number", "model_number",
    "upc_number", "additional_info", "notes", "check_in_procedure", "check_out_procedure",
    "icon_name", "suggested_vendor", "suggested_category", "suggested_location",
    "manufacturer_brand", "visible_condition", "is_vehicle", "vehicle_type", "license_plate",
    "image_paths", "original_filenames", "confidence_score", "flagged_for_review", "review_reason",
]

TOOL_CSV_COLUMNS = [
    "name", "description", "width", "height", "length", "depth", "weight", "value",
    "barcode_number", "serial_number", "reference_number", "model_number", "tool_number",
    "additional_info", "notes", "suggested_vendor", "suggested_category",
    "manufacturer_brand",
    "image_paths", "original_filenames", "confidence_score", "flagged_for_review", "review_reason",
]

PART_CSV_COLUMNS = [
    "name", "description", "serial_number", "reference_number", "model_number",
    "part_number", "additional_info", "notes", "value",
    "suggested_vendor", "suggested_category", "manufacturer_brand",
    "image_paths", "original_filenames", "confidence_score", "flagged_for_review", "review_reason",
]

CHEMICAL_CSV_COLUMNS = [
    "name", "description", "chemical_formula", "cas_number", "ec_number", "un_number",
    "ghs_hazard_class", "signal_word", "physical_state", "color", "odor",
    "ph", "melting_point", "boiling_point", "flash_point", "solubility",
    "storage_class", "storage_requirements",
    "respiratory_protection", "hand_protection", "eye_protection", "skin_protection",
    "first_aid_measures", "firefighting_measures", "spill_leak_procedures", "disposal_considerations",
    "unit_of_measure", "hazard_statements", "precautionary_statements",
    "manufacturer_name", "suggested_vendor", "suggested_category",
    "image_paths", "original_filenames", "confidence_score", "flagged_for_review", "review_reason",
]

UNCLASSIFIED_CSV_COLUMNS = [
    "original_filename", "image_path", "confidence_score",
    "flagged_for_review", "review_reason", "classification_reasoning",
]

CSV_COLUMNS_BY_TYPE = {
    EntityType.ASSET: ASSET_CSV_COLUMNS,
    EntityType.TOOL: TOOL_CSV_COLUMNS,
    EntityType.PART: PART_CSV_COLUMNS,
    EntityType.CHEMICAL: CHEMICAL_CSV_COLUMNS,
    EntityType.UNCLASSIFIED: UNCLASSIFIED_CSV_COLUMNS,
}
