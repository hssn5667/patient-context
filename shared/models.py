"""
Shared Pydantic models for MediTwin AI
These models define the data contracts between all agents
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import re

# ----- patient State classes -----

class Demographics(BaseModel):
    """Patient demographic information"""
    name: str
    age: int
    gender: str
    dob: str  # ISO 8601 date


class Condition(BaseModel):
    """FHIR Condition resource simplified"""
    code: str  # ICD-10 code
    display: str
    onset: Optional[str] = None  # ISO 8601 date


class Medication(BaseModel):
    """Current medication"""
    drug: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    status: str = "active"


class Allergy(BaseModel):
    """Allergy or intolerance"""
    substance: str
    reaction: Optional[str] = None
    severity: Optional[str] = None


class LabResult(BaseModel):
    """Lab observation"""
    loinc: str  # LOINC code
    display: str
    value: float
    unit: str
    reference_high: Optional[float] = None
    reference_low: Optional[float] = None
    flag: Optional[str] = None  # NORMAL, HIGH, LOW, CRITICAL


class DiagnosticReport(BaseModel):
    """Diagnostic report summary"""
    code: str
    display: str
    conclusion: Optional[str] = None
    issued: Optional[str] = None


class Encounter(BaseModel):
    """Recent encounter"""
    id: str
    type: str
    date: str
    reason: Optional[str] = None


# ----- Patient State -----

class PatientState(BaseModel):
    """
    The canonical patient state object.
    This is the contract between Patient Context Agent and all downstream agents.
    
    CRITICAL: Changes to this schema break all agents. Treat as versioned API.
    """
    patient_id: str
    demographics: Demographics
    active_conditions: List[Condition] = Field(default_factory=list)
    medications: List[Medication] = Field(default_factory=list)
    allergies: List[Allergy] = Field(default_factory=list)
    lab_results: List[LabResult] = Field(default_factory=list)
    diagnostic_reports: List[DiagnosticReport] = Field(default_factory=list)
    recent_encounters: List[Encounter] = Field(default_factory=list)
    state_timestamp: str  # ISO 8601
    imaging_available: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "example-patient-123",
                "demographics": {
                    "name": "John Doe",
                    "age": 54,
                    "gender": "male",
                    "dob": "1970-03-14"
                },
                "active_conditions": [
                    {
                        "code": "J18.9",
                        "display": "Pneumonia, unspecified",
                        "onset": "2025-04-01"
                    }
                ],
                "medications": [
                    {
                        "drug": "Amoxicillin",
                        "dose": "500mg",
                        "frequency": "TID",
                        "status": "active"
                    }
                ],
                "allergies": [
                    {
                        "substance": "Penicillin",
                        "reaction": "Anaphylaxis",
                        "severity": "severe"
                    }
                ],
                "lab_results": [
                    {
                        "loinc": "26464-8",
                        "display": "WBC",
                        "value": 14.2,
                        "unit": "10*3/uL",
                        "reference_high": 11.0,
                        "flag": "HIGH"
                    }
                ],
                "diagnostic_reports": [],
                "recent_encounters": [],
                "state_timestamp": "2025-04-01T10:30:00Z",
                "imaging_available": False
            }
        }
