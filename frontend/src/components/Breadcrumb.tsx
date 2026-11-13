import { Link } from "react-router-dom";

export default function Breadcrumb({
  items,
}: {

  items: { label: string; to?: string }[];
}) {
  return (
    <nav className="flex flex-wrap items-center gap-2 text-xs text-ink-muted">
      {items.map((item, i) => (
        <span key={i} className="flex items-center gap-2">
          {i > 0 && <span aria-hidden>/</span>}
          {item.to ? (
            <Link to={item.to} className="hover:text-ink">
              {item.label}
            </Link>
          ) : (
            <span>{item.label}</span>
          )}
        </span>
      ))}
    </nav>
  );
}
