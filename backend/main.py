from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import database and models
from database import Base, engine
from models import User, Patient, Assessment

# Import routers
from routes_auth import router as auth_router
from routes_patients import router as patients_router
from routes_assessments import router as assessments_router
from routes_websocket import router as websocket_router

# Create database tables (only if database is configured)
# NOTE: Update DATABASE_URL in .env with your Supabase connection string
try:
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created/verified")
except Exception as e:
    print(f"⚠️  Database not connected: {e}")
    print("📝 Update DATABASE_URL in .env with your Supabase connection string")
    print("   Get it from: Supabase Dashboard → Settings → Database → Connection String")

app = FastAPI(
    title="NeuroBalance AI Backend",
    version="1.0.0",
    description="AI-driven psychosomatic assessment platform backend API",
    docs_url="/docs",  # Interactive API documentation
    redoc_url="/redoc"  # Alternative documentation view
)

# Configure CORS - allows frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# This connects all route files to the main app
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(patients_router, prefix="/patients", tags=["Patients"])
app.include_router(assessments_router, prefix="/assessments", tags=["Assessments"])
app.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])


@app.get("/")
def root():
    """Root endpoint - basic health check"""
    return {
        "message": "NeuroBalance AI Backend API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "NeuroBalance AI Backend"
    }


if __name__ == "__main__":
    import uvicorn
    # Run server on localhost:8000 with auto-reload for development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-restart on code changes
    )
