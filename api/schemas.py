from pydantic import BaseModel
from typing import Optional, List, Dict


class AssessResponse(BaseModel):
    final_score: float
    band: str
    breakdown: Dict[str, float]
    recommendations: List[str]
    crisis_alert: Optional[str] = None
    empathetic_summary: Optional[str] = None


class SaveSessionRequest(BaseModel):
    score_result: dict
    feedback: Optional[dict] = None


class TrendResponse(BaseModel):
    direction: str
    delta: Optional[float] = None
    sessions_logged: int
    latest_score: Optional[float] = None
    prior_avg: Optional[float] = None
