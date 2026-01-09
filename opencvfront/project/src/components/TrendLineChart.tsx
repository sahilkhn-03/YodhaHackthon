import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

interface TrendLineChartProps {
  data: number[];
}

export function TrendLineChart({ data }: TrendLineChartProps) {
  const labels = data.map((_, i) => {
    const secondsAgo = (data.length - 1 - i) * 2;
    return secondsAgo === 0 ? 'Now' : `-${secondsAgo}s`;
  });

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Stress Level',
        data,
        borderColor: 'rgb(75, 85, 99)',
        backgroundColor: 'rgba(75, 85, 99, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 4,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: true,
        mode: 'index' as const,
        intersect: false,
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
        ticks: {
          color: 'rgb(156, 163, 175)',
          font: {
            size: 11,
          },
        },
      },
      y: {
        min: 0,
        max: 100,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
        ticks: {
          color: 'rgb(156, 163, 175)',
          font: {
            size: 11,
          },
        },
      },
    },
    interaction: {
      mode: 'nearest' as const,
      axis: 'x' as const,
      intersect: false,
    },
  };

  return (
    <div className="bg-white rounded-lg p-6 border border-gray-200">
      <h3 className="text-sm font-medium text-gray-700 mb-4">Stress Trend (Last 2 Minutes)</h3>
      <div className="h-48">
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}
