import { Circle } from 'lucide-react';
import { useEffect, useState } from 'react';

interface HeaderProps {
  isConnected: boolean;
}

export function Header({ isConnected }: HeaderProps) {
  const [sessionTime, setSessionTime] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setSessionTime((prev) => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <header className="fixed top-0 left-0 right-0 bg-white border-b border-gray-200 z-50">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <h1 className="text-2xl font-light text-gray-800">
           NeuroBalance <span className="font-normal text-gray-900">AI</span>
        </h1>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <Circle
              size={12}
              className={`transition-colors duration-300 ${
                  isConnected
                    ? 'fill-gray-700 text-gray-700 animate-pulse'
                    : 'fill-gray-400 text-gray-400'
              }`}
            />
            <span className="text-sm text-gray-600">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          <div className="text-sm text-gray-500">
            Session: <span className="font-medium text-gray-700">{formatTime(sessionTime)}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
