import React, { useEffect, useRef } from 'react';
import { FaceMesh, type NormalizedLandmarkList } from '@mediapipe/face_mesh';
import { computeFaceMetrics } from '../lib/faceMetrics';

interface CameraDisplayProps {
  isCameraEnabled: boolean;
  onCameraToggle: () => void;
  onMetricsUpdate?: (metrics: {
    eye_openness: number;
    brow_tension: number;
    jaw_tension: number;
    facial_asymmetry: number;
    head_motion: number;
    facial_stress_score: number;
  }) => void;
}

export function CameraDisplay({ isCameraEnabled, onCameraToggle, onMetricsUpdate }: CameraDisplayProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const faceMeshRef = useRef<FaceMesh | null>(null);
  const animationRef = useRef<number | null>(null);
  const prevMetricsRef = useRef<ReturnType<typeof computeFaceMetrics> | undefined>(undefined);

  // Initialize FaceMesh when camera first turns on
  useEffect(() => {
    let stream: MediaStream | null = null;
    let running = false;

    async function startCamera() {
      if (!videoRef.current) return;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        // Setup MediaPipe FaceMesh
        const faceMesh = new FaceMesh({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
        });
        faceMesh.setOptions({
          maxNumFaces: 1,
          refineLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        faceMesh.onResults((results) => {
          const landmarks = results.multiFaceLandmarks?.[0] as NormalizedLandmarkList | undefined;
          if (landmarks) {
            const metrics = computeFaceMetrics(landmarks, prevMetricsRef.current);
            prevMetricsRef.current = metrics;
            onMetricsUpdate?.(metrics);
          }
        });
        faceMeshRef.current = faceMesh;

        running = true;
        renderLoop();
      } catch (err) {
        console.error('Camera start error', err);
      }
    }

    async function renderLoop() {
      if (!running) return;
      if (videoRef.current && faceMeshRef.current) {
        await faceMeshRef.current.send({ image: videoRef.current });
      }
      animationRef.current = requestAnimationFrame(renderLoop);
    }

    function stopCamera() {
      running = false;
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
      }
    }

    if (isCameraEnabled) {
      startCamera();
    }

    return () => {
      stopCamera();
    };
    // Only respond to isCameraEnabled changes
  }, [isCameraEnabled, onMetricsUpdate]);

  return (
    <section className="relative rounded-2xl border border-gray-300 bg-gray-200 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-300 bg-gray-100">
        <h2 className="text-sm font-medium text-gray-700">Camera</h2>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onCameraToggle}
            className={`text-xs px-3 py-1 rounded-md border transition-colors ${
              isCameraEnabled
                ? 'bg-gray-800 text-white border-gray-800'
                : 'bg-white text-gray-700 border-gray-300 hover:border-gray-400'
            }`}
          >
            Camera: {isCameraEnabled ? 'On' : 'Off'}
          </button>
        </div>
      </div>

      {/* Square camera area */}
      <div className="relative w-full max-w-md mx-auto aspect-square">
        {/* Placeholder camera surface */}
        {!isCameraEnabled && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-sm text-gray-600">Camera off</span>
          </div>
        )}
        {isCameraEnabled && (
          <>
            <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover" muted playsInline />
          </>
        )}
      </div>
    </section>
  );
}
