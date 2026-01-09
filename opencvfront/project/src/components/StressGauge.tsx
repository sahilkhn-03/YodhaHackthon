import { useEffect, useState } from 'react';

interface StressGaugeProps {
  value: number;
}

export function StressGauge({ value }: StressGaugeProps) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const duration = 500;
    const steps = 30;
    const increment = (value - displayValue) / steps;
    let currentStep = 0;

    const interval = setInterval(() => {
      currentStep++;
      if (currentStep >= steps) {
        setDisplayValue(value);
        clearInterval(interval);
      } else {
        setDisplayValue((prev) => prev + increment);
      }
    }, duration / steps);

    return () => clearInterval(interval);
  }, [value]);

  const getColorClass = () => {
    if (displayValue < 40) return 'text-gray-700';
    if (displayValue < 70) return 'text-gray-600';
    return 'text-gray-500';
  };

  const getBgClass = () => {
    if (displayValue < 40) return 'bg-gray-50';
    if (displayValue < 70) return 'bg-gray-100';
    return 'bg-gray-200';
  };

  return (
    <div
      className={`rounded-2xl p-8 transition-colors duration-500 ${getBgClass()} border-2 border-gray-100`}
    >
      <div className="text-center">
        <div className={`text-7xl font-light transition-colors duration-500 ${getColorClass()}`}>
          {Math.round(displayValue)}
        </div>
        <div className="text-sm text-gray-500 mt-2 uppercase tracking-wide">
          Facial Stress Indicator
        </div>
        <div className="mt-4 flex justify-center gap-4 text-xs text-gray-400">
          <span>Low <span className="text-gray-600">(&lt;40)</span></span>
          <span>Moderate <span className="text-gray-500">(40-70)</span></span>
          <span>Elevated <span className="text-gray-400">(&gt;70)</span></span>
        </div>
      </div>
    </div>
  );
}
