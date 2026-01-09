"""
Authentication routes: register, login, get current user.

Why separate route files?
- Organization: Keep related endpoints together
- Scalability: Easy to add new routes without cluttering main.py
- Team collaboration: Different developers can work on different route files
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta

from database import get_db
from models import User
from schemas import UserCreate, UserLogin, UserResponse, Token
from auth import hash_password, verify_password, create_access_token, decode_token
from config import settings

# Create router (like a mini FastAPI app)
router = APIRouter()

# OAuth2 scheme for token authentication
# This tells FastAPI to look for "Authorization: Bearer <token>" header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user (clinician or admin).
    
    Why this endpoint?
    - Allows new clinicians to create accounts
    - Validates email format automatically (EmailStr in schema)
    - Hashes password before storing (never store plain text!)
    
    Flow:
    1. Frontend sends: email, username, password
    2. Check if user already exists → reject if duplicate
    3. Hash password
    4. Save to database
    5. Return user info (without password!)
    """
    # Check if username or email already exists
    existing_user = db.query(User).filter(
        (User.email == user.email) | (User.username == user.username)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists"
        )
    
    # Create new user with hashed password
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=hash_password(user.password),  # Never store plain password!
        full_name=user.full_name,
        role=user.role
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)  # Get the user with auto-generated ID
    
    return db_user


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Login and get JWT access token.
    
    Why JWT?
    - Stateless: No server-side session storage needed
    - Frontend stores token, sends with each request
    - Token expires after 30 minutes (configurable)
    
    Flow:
    1. User enters username and password
    2. Look up user in database
    3. Verify password matches hashed password
    4. Generate JWT token with user info
    5. Return token to frontend
    6. Frontend stores token and sends it with each API request
    """
    # Find user by username
    user = db.query(User).filter(User.username == form_data.username).first()
    
    # Verify user exists and password is correct
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user account is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    This is a DEPENDENCY function used in protected routes.
    
    How to use in routes:
        @router.get("/protected")
        def protected_route(current_user: User = Depends(get_current_user)):
            # current_user is automatically populated from token
            return {"user": current_user.username}
    
    Flow:
    1. Extract token from "Authorization: Bearer <token>" header
    2. Decode token to get user_id
    3. Look up user in database
    4. Return user object (or raise 401 if invalid)
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Decode token
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception
    
    user_id: int = payload.get("user_id")
    if user_id is None:
        raise credentials_exception
    
    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """
    Get current user's profile.
    
    Protected route - requires valid JWT token.
    
    Frontend usage:
        fetch('/auth/me', {
            headers: { 'Authorization': `Bearer ${token}` }
        })
    """
    return current_user


# Route explanation summary:
#
# POST /auth/register
# - Public endpoint (no token required)
# - Creates new user account
# - Returns user data
#
# POST /auth/login
# - Public endpoint
# - Validates credentials
# - Returns JWT token
#
# GET /auth/me
# - Protected endpoint (requires token)
# - Returns current user's profile
# - Uses get_current_user dependency
