import os
from fastapi import APIRouter, Header, HTTPException, Query
from src.kb.kb_service import fetch_from_kb
from src.security.token_manager import generate_token, validate_token
from dotenv import load_dotenv

# Initialize the api router
router = APIRouter()

load_dotenv()


def _get_internal_api_key() -> str:
    return os.getenv("INTERNAL_API_KEY", "secret-key").strip()


def _get_token_expiry_seconds() -> int:
    return int(os.getenv("TOKEN_EXPIRY_SECONDS", "60"))


@router.post("/kb/token")
def create_kb_token(x_api_key: str = Header(None, alias="X-API-KEY")):
    if (x_api_key or "").strip() != _get_internal_api_key():
        raise HTTPException(status_code=401, detail="Unauthorized API key")

    token = generate_token()
    return {"token": token, "expires_in": _get_token_expiry_seconds()}


@router.post("/kb/fetch")
def fetch_kb_data(query: str = Query(...), authorization: str = Header(None, alias="Authorization")):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Access Denied")

    token = authorization.split(" ", 1)[1].strip()
    if not validate_token(token):
        raise HTTPException(status_code=401, detail="Access Denied")

    data = fetch_from_kb(query)
    if data is None:
        return {"data": None, "message": "No KB match for query"}

    return {"data": data}
