"""
Authentication utilities: password hashing and JWT token generation.

Why JWT (JSON Web Tokens)?
- Stateless authentication (no server-side session storage)
- Secure: Token contains encrypted user info
- Scalable: Works across multiple servers
- Frontend can store token in localStorage and send with each request
"""

from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
from config import settings

# Password hashing context using bcrypt
# Why bcrypt? Industry-standard, slow by design (prevents brute force attacks)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    Hash a plain-text password using bcrypt.
    
    Why hash?
    - NEVER store passwords in plain text!
    - If database is compromised, passwords remain secure
    - bcrypt is slow, making brute-force attacks impractical
    
    Example:
        plain: "mypassword123"
        hashed: "$2b$12$KIXl3fG.../5Q2Fc6FHvGO" (irreversible)
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify if plain password matches the hashed password.
    
    How it works:
    1. User enters password during login
    2. We hash it with the same salt
    3. Compare with stored hash
    4. Return True if match, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """
    Create a JWT access token.
    
    What's in the token?
    - User info (username, user_id)
    - Expiration time (default: 30 minutes)
    - Signature (proves it wasn't tampered with)
    
    Flow:
    1. User logs in with username/password
    2. We verify credentials
    3. Generate JWT token
    4. Frontend stores token
    5. Frontend sends token with each API request
    6. Backend verifies token and identifies user
    
    Example token (3 parts separated by dots):
    eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.
    eyJzdWIiOiJkci5zYXJhaCIsImV4cCI6MTY3ODk4NzY1NH0.
    SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
    
    Decode to see contents (but can't modify without secret key!)
    """
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    # Encode with secret key (from .env)
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt


def decode_token(token: str) -> dict:
    """
    Decode and verify a JWT token.
    
    What this checks:
    1. Token signature is valid (not tampered)
    2. Token hasn't expired
    3. Token was signed with our secret key
    
    Returns user data if valid, raises exception if invalid.
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except jwt.JWTError:
        return None


# Authentication Flow Summary:
#
# REGISTRATION:
# 1. User submits: username, password, email
# 2. hash_password() → Store hashed password in database
#
# LOGIN:
# 1. User submits: username, password
# 2. Look up user in database
# 3. verify_password() → Check if password matches
# 4. create_access_token() → Generate JWT token
# 5. Return token to frontend
#
# PROTECTED ROUTES:
# 1. Frontend sends token in header: "Authorization: Bearer <token>"
# 2. decode_token() → Verify and extract user info
# 3. Load user from database
# 4. Allow access to protected resource
