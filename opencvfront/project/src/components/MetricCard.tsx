import { LucideIcon } from 'lucide-react';
import { useEffect, useState } from 'react';

interface MetricCardProps {
  title: string;
  value: number;
  icon: LucideIcon;
  helperText?: string;
}

export function MetricCard({ title, value, icon: Icon, helperText }: MetricCardProps) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const duration = 400;
    const steps = 20;
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

  const percentage = Math.round(displayValue * 100);

  return (
    <div className="bg-white rounded-lg p-5 border-2 border-gray-200 hover:border-gray-300 transition-colors">
      <div className="flex items-center gap-3 mb-3">
        <Icon size={20} className="text-gray-600" />
        <h3 className="text-sm font-medium text-gray-700">{title}</h3>
      </div>

      <div className="mb-2">
        <div className="text-2xl font-light text-gray-800">{percentage}%</div>
      </div>

      <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
        <div
          className="h-full bg-gray-700 transition-all duration-300 ease-out rounded-full"
          style={{ width: `${percentage}%` }}
        />
      </div>

      {helperText && <div className="text-xs text-gray-400 mt-2">{helperText}</div>}
    </div>
  );
}
