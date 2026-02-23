from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class InferenceRequest(BaseModel):
    subject_id: str = Field(..., description="Unique subject/patient identifier")
    x_tab: List[float] = Field(default_factory=list, description="Tabular feature vector")
    tab_present: bool = Field(default=True, description="Whether tabular modality is present")
    x_sig: List[float] = Field(..., description="Flattened signal array of shape (C*T)")
    sig_channels: int = Field(..., description="C")
    sig_length: int = Field(..., description="T")
    sig_present: bool = Field(default=True, description="Whether signal modality is present")

class ExplanationStub(BaseModel):
    available: bool = False
    method: Optional[str] = None
    summary: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)

class InferenceResponse(BaseModel):
    subject_id: str
    risk_probabilities: List[float]
    predicted_risk_class: int
    confidence: float
    high_risk_flag: bool
    defer_to_clinician_flag: bool
    modality_attention: Optional[List[float]] = None
    modality_reliability: Optional[List[float]] = None
    explanations: ExplanationStub = Field(default_factory=ExplanationStub)
    provenance: Dict[str, Any] = Field(default_factory=dict)

class BatchInferenceRequest(BaseModel):
    items: List[InferenceRequest]

class CohortSummary(BaseModel):
    n: int
    high_risk_rate: float
    deferral_rate: float
    mean_confidence: float

class BatchInferenceResponse(BaseModel):
    results: List[InferenceResponse]
    cohort_summary: CohortSummary
    provenance: Dict[str, Any] = Field(default_factory=dict)
