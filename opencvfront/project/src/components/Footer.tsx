import { Info } from 'lucide-react';

export function Footer() {
  return (
    <footer className="mt-12 pb-8">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-center gap-2 text-xs text-gray-400">
          <Info size={14} />
          <span>Indicators reflect facial tension patterns, not medical diagnosis.</span>
        </div>
      </div>
    </footer>
  );
}
