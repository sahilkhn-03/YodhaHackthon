import { useEffect, useState } from 'react';
import { Eye, Frown, Gauge, Move } from 'lucide-react';
import { Header } from './components/Header';
import { CameraDisplay } from './components/CameraDisplay';
import { StressGauge } from './components/StressGauge';
import { MetricCard } from './components/MetricCard';
import { TrendLineChart } from './components/TrendLineChart';
import { Footer } from './components/Footer';

interface FaceData {
  timestamp: number;
  eye_openness: number;
  brow_tension: number;
  jaw_tension: number;
  facial_asymmetry: number;
  head_motion: number;
  facial_stress_score: number;
}

function App() {
  const [isConnected, setIsConnected] = useState(true);
  const [currentData, setCurrentData] = useState<FaceData>({
    timestamp: Date.now(),
    eye_openness: 0.8,
    brow_tension: 0.3,
    jaw_tension: 0.2,
    facial_asymmetry: 0.1,
    head_motion: 0.15,
    facial_stress_score: 25,
  });
  const [trendData, setTrendData] = useState<number[]>([25]);
  const [isCameraEnabled, setIsCameraEnabled] = useState(false);

  useEffect(() => {
    // Ensure the browser tab title is correctly set
    document.title = 'NeuroBalance AI';
  }, []);

  // Simulator: only run when camera is off
  useEffect(() => {
    if (isCameraEnabled) return;
    const interval = setInterval(() => {
      const newData: FaceData = {
        timestamp: Date.now(),
        eye_openness: Math.max(0.5, Math.min(1, currentData.eye_openness + (Math.random() - 0.5) * 0.1)),
        brow_tension: Math.max(0, Math.min(1, currentData.brow_tension + (Math.random() - 0.5) * 0.15)),
        jaw_tension: Math.max(0, Math.min(1, currentData.jaw_tension + (Math.random() - 0.5) * 0.15)),
        facial_asymmetry: Math.max(0, Math.min(1, currentData.facial_asymmetry + (Math.random() - 0.5) * 0.1)),
        head_motion: Math.max(0, Math.min(1, currentData.head_motion + (Math.random() - 0.5) * 0.12)),
        facial_stress_score: Math.max(
          0,
          Math.min(100, currentData.facial_stress_score + (Math.random() - 0.5) * 8)
        ),
      };

      setCurrentData(newData);

      setTrendData((prev) => {
        const updated = [...prev, newData.facial_stress_score];
        return updated.slice(-60);
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [currentData, isCameraEnabled]);

  return (
    <div className="min-h-screen bg-gray-50">
      <Header isConnected={isConnected} />

      <main className="pt-24 pb-8">
        <div className="max-w-7xl mx-auto px-6">
          {/* Camera section above Stress Gauge */}
          <div className="mb-6">
            <CameraDisplay
              isCameraEnabled={isCameraEnabled}
              onCameraToggle={() => setIsCameraEnabled((prev) => !prev)}
              onMetricsUpdate={(m) => {
                const newData: FaceData = {
                  timestamp: Date.now(),
                  eye_openness: m.eye_openness,
                  brow_tension: m.brow_tension,
                  jaw_tension: m.jaw_tension,
                  facial_asymmetry: m.facial_asymmetry,
                  head_motion: m.head_motion,
                  facial_stress_score: m.facial_stress_score,
                };
                setCurrentData(newData);
                setTrendData((prev) => [...prev, m.facial_stress_score].slice(-60));
              }}
            />
          </div>
          <div className="mb-8">
            <StressGauge value={currentData.facial_stress_score} />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <MetricCard
              title="Eye Openness"
              value={currentData.eye_openness}
              icon={Eye}
              helperText="Wider eyes may indicate alertness"
            />
            <MetricCard
              title="Brow Tension"
              value={currentData.brow_tension}
              icon={Frown}
              helperText="Furrowed brow suggests focus or concern"
            />
            <MetricCard
              title="Jaw Tension"
              value={currentData.jaw_tension}
              icon={Gauge}
              helperText="Clenched jaw may indicate stress"
            />
            <MetricCard
              title="Head Motion"
              value={currentData.head_motion}
              icon={Move}
              helperText="Movement patterns during session"
            />
          </div>

          <TrendLineChart data={trendData} />
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default App;
