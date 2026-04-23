import uuid
import time

TOKENS = {}

TOKEN_EXPIRY_SECONDS = 60


def generate_token():
    token = str(uuid.uuid4())
    expiry = time.time() + TOKEN_EXPIRY_SECONDS

    TOKENS[token] = expiry
    return token


def validate_token(token):
    if token not in TOKENS:
        return False

    if time.time() > TOKENS[token]:
        del TOKENS[token]
        return False

    return True