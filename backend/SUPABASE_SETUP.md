# Supabase Setup Instructions

## Get Your Supabase Connection String

1. **Go to Supabase Dashboard:** https://supabase.com/dashboard
2. **Create a new project** (or use existing)
3. **Navigate to:** Settings → Database → Connection String
4. **Copy the URI** (Connection Pooling recommended for serverless)

## Update .env File

Replace `DATABASE_URL` in `.env` with your Supabase connection string:

```
DATABASE_URL=postgresql://postgres.[YOUR-PROJECT-REF]:[YOUR-PASSWORD]@aws-0-[region].pooler.supabase.com:5432/postgres
```

## Privacy-First Data Model

### ✅ WHAT WE STORE (Analysis Results Only):

**Assessment Table:**
- Patient ID (anonymized)
- Timestamp
- **Stress score** (0-1 float)
- **Anxiety score** (0-1 float)
- **Heart rate metrics** (avg, min, max BPM)
- **Analysis summary** (text):
  - "Elevated heart rate detected (avg 95 BPM, baseline 70 BPM)"
  - "Likely reasons: stress response, anxiety indicators present"
- **Recommendations** (text):
  - "Consider relaxation techniques, monitor for 48 hours"
- Session duration
- Assessment type (initial/follow-up)

**Patient Table:**
- Patient ID (anonymized: PT-20260109-ABC123)
- Basic demographics (age range, no PII)
- Baseline metrics (resting heart rate)
- Assessment history (count, dates only)

### ❌ WHAT WE NEVER STORE (Privacy Protected):

- **NO raw video frames**
- **NO audio recordings**
- **NO facial landmark coordinates**
- **NO voice waveforms**
- **NO personally identifiable information (PII)**

All AI processing happens **client-side or edge** → only final metrics sent to backend.

## Minimal FastAPI Usage

Since we're privacy-first, FastAPI backend is used ONLY for:

1. **Authentication** - Secure user login
2. **Storing analysis results** - Stress scores, heart rate changes
3. **Retrieving patient history** - Past assessment summaries
4. **Generating reports** - Trend analysis over time

**NOT used for:**
- Streaming video/audio (handled client-side)
- Real-time facial analysis (processed in browser)
- Voice processing (processed locally)

## Quick Setup

```powershell
# Install Supabase client (optional, for direct queries)
pip install supabase

# Update requirements.txt
pip freeze > requirements.txt

# Test connection
python -c "from database import engine; print(engine.connect())"
```

## Example: What Gets Stored

```json
{
  "assessment_id": "AS-20260109-XYZ",
  "patient_id": "PT-20260109-ABC",
  "timestamp": "2026-01-09T14:30:00Z",
  "stress_score": 0.72,
  "anxiety_score": 0.65,
  "heart_rate_avg": 95,
  "heart_rate_baseline": 70,
  "analysis_summary": "Heart rate elevated by 35% above baseline. Vocal stress indicators present. Facial tension detected in forehead region.",
  "likely_reasons": "Acute stress response, possible anxiety trigger",
  "recommendations": "Deep breathing exercises, follow-up in 24 hours"
}
```

**Notice:** No images, no audio files, no raw biometric data—just the interpreted results.

## Security Features

- Row-level security (RLS) in Supabase
- Patient data encrypted at rest
- TLS for all connections
- Anonymized patient IDs
- No PII in logs or error messages
