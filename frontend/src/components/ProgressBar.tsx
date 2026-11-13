export default function ProgressBar({
  value,
  className = "",
}: {

  value: number;
  className?: string;
}) {
  const pct = Math.max(0, Math.min(100, value));
  return (

    <div
      className={`flex h-1 overflow-hidden rounded-full bg-hair ${className}`}
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className="h-full rounded-full bg-brand transition-[width]" style={{ width: `${pct}%` }} />
    </div>
  );
}
