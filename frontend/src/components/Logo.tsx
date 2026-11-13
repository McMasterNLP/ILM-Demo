export default function Logo({ className = "w-8 h-8" }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 32 32"
      className={className}
      role="img"
      aria-label="Ilm"
    >
      <rect width="32" height="32" rx="8" fill="#7A003C" />
      <path
        d="M8 23V10l8 8 8-8v13"
        fill="none"
        stroke="#ffffff"
        strokeWidth={2.4}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="8" cy="10" r="2" fill="#ffffff" />
      <circle cx="24" cy="10" r="2" fill="#ffffff" />
      <circle cx="16" cy="18" r="2" fill="#ffffff" fillOpacity={0.85} />
    </svg>
  );
}
