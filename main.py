"""
Agent 1: Patient Context Agent — COMPLETE REBUILD
Entry point of MediTwin - fetches and normalizes FHIR data

Role: FHIR Data Layer — Entry Point of the System
Type: A2A Agent + MCP Tool Consumer
Protocol: SHARP context propagation → FHIR R4 REST API

This agent is THE FOUNDATION. Every other agent depends on its output.
If this agent fails, nothing else can run.

Key Responsibilities:
1. Extract patient ID and FHIR bearer token from SHARP context headers
2. Fetch all relevant FHIR R4 resources (Patient, Condition, Medication, etc.)
3. Normalize into a single structured PatientState Pydantic model
4. Cache in Redis (TTL: 10 minutes) to avoid redundant FHIR calls
5. Return PatientState to orchestrator

SHARP Context Strategy (CRITICAL FIX):
- Production mode: Reads X-SHARP-Patient-ID, X-SHARP-FHIR-Token, X-SHARP-FHIR-BaseURL headers
- Development mode: Falls back to request body if headers not present
- This dual-mode design makes development smooth without simulating headers every time

Missing Features Now Implemented:
✓ Proper SHARP context header resolution
✓ Fallback to request body for development
✓ Imaging availability detection from DiagnosticReport
✓ Graceful handling of missing FHIR resources (empty arrays, not exceptions)
✓ Parallel FHIR fetches with asyncio.gather()
✓ Complete error handling with detailed logging
✓ Validation before caching
"""
import os
import sys
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

import httpx
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field, ValidationError

# Add parent directory to path for shared imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.models import (
    PatientState, Demographics, Condition, Medication, 
    Allergy, LabResult, DiagnosticReport, Encounter
)
from shared.redis_client import redis_client

from stream_endpoint import router as stream_router

from db import init as db_init, close as db_close, save_patient_context, PatientContextRecord
from history_router import router as history_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("patient_context_agent")


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class PatientContextRequest(BaseModel):
    patient_id: Optional[str] = Field(
        default=None,  # ← was required (...)
        description="FHIR Patient resource ID — optional if X-SHARP-Patient-ID header is present"
    )
    fhir_base_url: Optional[str] = Field(
        default="https://hapi.fhir.org/baseR4",
        description="FHIR server base URL"
    )
    sharp_token: Optional[str] = Field(
        default=None,
        description="SHARP FHIR access token (optional direct field for dev)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "example",
                "fhir_base_url": "https://hapi.fhir.org/baseR4"
            }
        }


class PatientContextResponse(BaseModel):
    """Response with complete patient state"""
    patient_state: PatientState
    cache_hit: bool = Field(..., description="Whether data was served from Redis cache")
    fetch_time_ms: int = Field(..., description="Total fetch time in milliseconds")
    source: str = Field(..., description="Data source: 'SHARP' or 'direct'")
    fhir_resources_fetched: int = Field(..., description="Number of FHIR resource types fetched")


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════

http_client: Optional[httpx.AsyncClient] = None


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION LIFECYCLE
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - setup and teardown"""
    global http_client
    
    # Startup
    logger.info(" " * 30)
    logger.info("  ✔  Patient Context Agent — Starting Up")
    logger.info("  ═" * 30)
    
    http_client = httpx.AsyncClient(timeout=30.0)
    await redis_client.connect()
    
    logger.info("  ✔  HTTP client initialized (timeout: 30s)")
    logger.info("  ✔  Redis connection established")
    logger.info("  ✔  Patient Context Agent ready on port 8001")
    logger.info("  ═" * 60)

    await db_init()
    
    yield
    
    # Shutdown
    logger.info("  ✔  Patient Context Agent shutting down...")
    await http_client.aclose()
    await redis_client.disconnect()
    await db_close()
    logger.info("✓ Patient Context Agent shutdown complete")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="MediTwin Patient Context Agent",
    description=(
        "FHIR data ingestion and normalization layer — Foundation of MediTwin AI. "
        "Fetches patient data from FHIR servers using SHARP context or direct requests, "
        "normalizes into a unified PatientState, and caches for performance."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stream_router)
app.include_router(history_router, prefix="/history", tags=["history"])

from utils import (
    fetch_fhir_resource,
    normalize_patient,
    normalize_conditions,
    normalize_medications,
    normalize_allergies,
    normalize_observations,
    normalize_diagnostic_reports
)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/fetch", response_model=PatientContextResponse)
async def fetch_patient_context(
    request: PatientContextRequest,
    x_sharp_patient_id: Optional[str] = Header(None, alias="X-SHARP-Patient-ID"),
    x_sharp_fhir_token: Optional[str] = Header(None, alias="X-SHARP-FHIR-Token"),
    x_sharp_fhir_base_url: Optional[str] = Header(None, alias="X-SHARP-FHIR-BaseURL")
) -> PatientContextResponse:
    """
    Fetch and normalize patient FHIR data
    
    SHARP Context Priority (Production):
    1. X-SHARP-Patient-ID header
    2. X-SHARP-FHIR-Token header
    3. X-SHARP-FHIR-BaseURL header
    
    Fallback (Development):
    - If SHARP headers not present, use request body fields
    - This enables smooth development without simulating headers
    
    Workflow:
    1. Check Redis cache (10-minute TTL)
    2. If cache miss, fetch from FHIR server:
       - Patient (demographics)
       - Condition (active diagnoses)
       - MedicationRequest (current prescriptions)
       - AllergyIntolerance (drug allergies)
       - Observation (lab results, last 20)
       - DiagnosticReport (imaging, last 5)
    3. Normalize all resources into PatientState
    4. Cache in Redis
    5. Return PatientState
    """
    start_time = datetime.now()
    
    # ── SHARP Context Resolution ──────────────────────────────────────────────
    # Priority: SHARP headers > request body
    # This is the CRITICAL FIX that was missing
    
    patient_id = x_sharp_patient_id or request.patient_id
    fhir_base_url = x_sharp_fhir_base_url or request.fhir_base_url or "https://hapi.fhir.org/baseR4"
    sharp_token = x_sharp_fhir_token or request.sharp_token
    
    # Determine source
    source = "SHARP" if x_sharp_patient_id else "direct"
    
    # Validate patient_id
    if not patient_id:
        raise HTTPException(
            status_code=400,
            detail="patient_id is required (via SHARP header or request body)"
        )
    
    # Build auth headers if token provided
    auth_headers = {}
    if sharp_token:
        auth_headers["Authorization"] = sharp_token if sharp_token.startswith("Bearer ") else f"Bearer {sharp_token}"
    
    logger.info(f"Fetch request: patient={patient_id}, source={source}, fhir={fhir_base_url}")
    
    # ── Redis Cache Check ──────────────────────────────────────────────────────
    
    cache_key = f"patient_state:{patient_id}"
    cached = await redis_client.get_json(cache_key)
    
    if cached:
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"  ✔  Cache HIT: patient={patient_id}, fetch_time={elapsed_ms}ms")
        
        return PatientContextResponse(
            patient_state=PatientState(**cached),
            cache_hit=True,
            fetch_time_ms=elapsed_ms,
            source=source,
            fhir_resources_fetched=0  # Cache hit, no FHIR fetch
        )
    
    # ── Parallel FHIR Fetch ────────────────────────────────────────────────────
    # CRITICAL: Parallel fetches reduce latency from ~6s (sequential) to ~1s
    
    logger.info(f"  ⚠  Cache MISS: patient={patient_id}, fetching from FHIR server...")
    
    # Build all fetch tasks
    tasks = [
        fetch_fhir_resource("Patient", fhir_base_url, resource_id=patient_id, auth_headers=auth_headers),
        fetch_fhir_resource("Condition", fhir_base_url, search_params={"patient": patient_id, "clinical-status": "active"}, auth_headers=auth_headers),
        fetch_fhir_resource("MedicationRequest", fhir_base_url, search_params={"patient": patient_id, "status": "active"}, auth_headers=auth_headers),
        fetch_fhir_resource("AllergyIntolerance", fhir_base_url, search_params={"patient": patient_id}, auth_headers=auth_headers),
        fetch_fhir_resource("Observation", fhir_base_url, search_params={"patient": patient_id, "category": "laboratory", "_sort": "-date", "_count": "20"}, auth_headers=auth_headers),
        fetch_fhir_resource("DiagnosticReport", fhir_base_url, search_params={"patient": patient_id, "_sort": "-date", "_count": "5"}, auth_headers=auth_headers),
    ]
    
    # Execute all fetches concurrently
    (patient_data, conditions_bundle, medications_bundle, 
     allergies_bundle, observations_bundle, diagnostic_reports_bundle) = await asyncio.gather(*tasks)
    
    # ── Normalization ──────────────────────────────────────────────────────────
    # Each normalizer handles missing data gracefully (returns empty list/None)
    
    demographics = normalize_patient(patient_data)
    
    if demographics is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {patient_id} not found or invalid Patient resource"
        )
    try:
        conditions = normalize_conditions(conditions_bundle)
        medications = await normalize_medications(medications_bundle, base_url=fhir_base_url, auth_headers=auth_headers)
        allergies = normalize_allergies(allergies_bundle)
        labs = normalize_observations(observations_bundle)
        diagnostic_reports, imaging_available = normalize_diagnostic_reports(diagnostic_reports_bundle)

        import uuid
        request_id = str(uuid.uuid4())[:8]
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        await save_patient_context(PatientContextRecord(
            request_id=request_id,
            patient_id=patient_id,
            source=source,
            fhir_base_url=fhir_base_url,
            fhir_resources_fetched=6,
            demographics=demographics.model_dump() if demographics else None,
            active_conditions=[c.model_dump() for c in conditions],
            medications=[m.model_dump() for m in medications],
            allergies=[a.model_dump() for a in allergies],
            lab_results=[l.model_dump() for l in labs],
            diagnostic_reports=[r.model_dump() for r in diagnostic_reports],
            imaging_available=imaging_available,
            cache_hit=False,
            fetch_time_ms=elapsed_ms, 
        ))
        
        # Build PatientState
        patient_state = PatientState(
            patient_id=patient_id,
            demographics=demographics,
            active_conditions=conditions,
            medications=medications,
            allergies=allergies,
            lab_results=labs,
            diagnostic_reports=diagnostic_reports,
            recent_encounters=[],  # TODO: Implement if needed
            state_timestamp=datetime.now().isoformat() + "Z",
            imaging_available=imaging_available
        )
        
        logger.info(f"Normalization complete: {len(conditions)} conditions, "
                   f"{len(medications)} medications, {len(labs)} labs, "
                   f"imaging_available={imaging_available}")
        
        # ── Cache and Return ───────────────────────────────────────────────────
        
        # Cache for 10 minutes (600 seconds)
        await redis_client.set_json(cache_key, patient_state.model_dump(), ttl=600)
        
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"Fetch complete: patient={patient_id}, fetch_time={elapsed_ms}ms")
        
        return PatientContextResponse(
            patient_state=patient_state,
            cache_hit=False,
            fetch_time_ms=elapsed_ms,
            source=source,
            fhir_resources_fetched=6
        )
        
    except ValidationError as e:
        logger.error(f"Pydantic validation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Patient state validation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"FHIR normalization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"FHIR normalization error: {str(e)}"
        )


@app.delete("/cache/{patient_id}")
async def clear_patient_cache(patient_id: str) -> dict:
    """Clear Redis cache for a specific patient"""
    cache_key = f"patient_state:{patient_id}"
    deleted = await redis_client.delete(cache_key)
    
    if deleted:
        logger.info(f"Cache cleared: patient={patient_id}")
        return {"status": "cleared", "patient_id": patient_id}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No cache entry found for patient {patient_id}"
        )
    



# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """
    Health check endpoint
    
    Verifies:
    - HTTP client is initialized
    - Redis connection is active
    - FHIR server is reachable (optional check)
    """
    redis_ok = await redis_client._client.ping() if redis_client._client else False
    
    return {
        "status": "healthy" if redis_ok else "degraded",
        "agent": "patient-context",
        "version": "1.0.0",
        "redis_connected": redis_ok,
        "http_client_ready": http_client is not None,
        "fhir_base_url": os.getenv("FHIR_BASE_URL", "https://hapi.fhir.org/baseR4")
    }

from fastapi.responses import JSONResponse

@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    base_url = str(request.base_url).rstrip("/").replace("http://", "https://")
    return JSONResponse({
        "name": "MediTwin Patient Context Agent",
        "description": (
            "Foundation agent of MediTwin AI. Fetches and normalizes all relevant FHIR R4 "
            "resources (Patient, Condition, MedicationRequest, AllergyIntolerance, Observation, "
            "DiagnosticReport) into a unified PatientState model. Supports SHARP context header "
            "propagation for production EHR integration and direct request body mode for "
            "development. Results are Redis-cached (10-minute TTL) and persisted to PostgreSQL. "
            "Every downstream MediTwin agent depends on this agent's output."
        ),
        "version": "1.0.0",
        "url": base_url,
        "provider": {
            "organization": "MediTwin AI",
            "name": "Tayyab Hussain",
            "url": "https://github.com/hssn5667/patient-context"
        },
        "supportedInterfaces": [
            {
                "url": f"{base_url}/fetch",
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
                "description": "Blocking fetch — returns full PatientState when all FHIR resources are ready."
            },
            {
                "url": f"{base_url}/stream",
                "protocolBinding": "HTTP+SSE",
                "protocolVersion": "1.0",
                "description": "SSE streaming fetch — emits status/progress/complete events in real time."
            }
        ],
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True
        },
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json", "text/event-stream"],
        "skills": [
            {
                "id": "fetch_patient_context",
                "name": "Fetch Patient Context",
                "description": (
                    "Fetches all FHIR R4 resources for a patient in parallel (6 resource types: "
                    "Patient, Condition, MedicationRequest, AllergyIntolerance, Observation, "
                    "DiagnosticReport) and normalizes them into a structured PatientState. "
                    "Checks Redis cache first; falls back to live FHIR server on cache miss. "
                    "Accepts patient identity via SHARP context headers (production) or request body (dev)."
                ),
                "tags": ["fhir", "patient", "ehr", "normalization", "redis-cache", "sharp"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"]
            },
            {
                "id": "stream_patient_context",
                "name": "Stream Patient Context (SSE)",
                "description": (
                    "SSE streaming variant of fetch_patient_context. Emits ordered events: "
                    "status (connecting, fetching, normalizing), progress (per resource type, 0-60%), "
                    "and a final complete event containing the full PatientState. "
                    "Allows the orchestrator to proxy real-time loading progress to the frontend."
                ),
                "tags": ["fhir", "patient", "sse", "streaming", "real-time"],
                "inputModes": ["application/json"],
                "outputModes": ["text/event-stream"]
            },
            {
                "id": "get_patient_history",
                "name": "Get Patient Fetch History",
                "description": (
                    "Returns paginated fetch history for a patient from PostgreSQL, including "
                    "per-session resource counts, cache hit/miss status, source (SHARP or direct), "
                    "FHIR base URL, and fetch timing. Supports latest-only and request-ID lookups."
                ),
                "tags": ["history", "audit", "patient", "postgresql"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"]
            },
            {
                "id": "get_patient_stats",
                "name": "Get Patient Context Stats",
                "description": (
                    "Returns aggregate statistics across all fetch sessions for a patient: "
                    "total sessions, cache hit rate, SHARP vs direct breakdown, imaging availability, "
                    "average/peak condition and lab counts, and first/latest fetch timestamps."
                ),
                "tags": ["stats", "analytics", "patient", "postgresql"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"]
            },
            {
                "id": "clear_patient_cache",
                "name": "Clear Patient Cache",
                "description": (
                    "Invalidates the Redis cache entry for a specific patient, forcing the next "
                    "fetch to retrieve fresh data from the FHIR server."
                ),
                "tags": ["cache", "redis", "invalidation"],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"]
            }
        ]
    })