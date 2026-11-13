import { useLang } from "../lib/i18n";
import { ApiError } from "../lib/http";
import { RefreshIcon, XCircleIcon } from "./Icons";

export function LoadingCards({ count = 6 }: { count?: number }) {
  return (
    <div className="mt-6 grid gap-5 md:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="h-56 animate-pulse rounded-xl border border-hair bg-white/60" />
      ))}
    </div>
  );
}

export function LoadingPanel() {
  const { t } = useLang();
  return (
    <div className="card mt-6 p-10 text-center text-sm text-ink-muted" role="status">
      {t("loading")}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: Error; onRetry?: () => void }) {
  const { t } = useLang();

  const gated = error instanceof ApiError && error.status === 403;

  return (
    <div className="card mt-6 p-8 text-center">
      {!gated && <XCircleIcon className="mx-auto h-7 w-7 text-brand" />}
      <p className="mt-3 text-sm font-medium">{gated ? t("quizLockedHint") : t("errorTitle")}</p>
      <p className="mt-1.5 text-xs text-ink-muted">{error.message}</p>
      {onRetry && !gated && (
        <button type="button" className="btn-quiet mx-auto mt-5" onClick={onRetry}>
          <RefreshIcon className="h-4 w-4" />
          {t("retry")}
        </button>
      )}
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return <p className="card mt-6 px-5 py-8 text-center text-sm text-ink-muted">{message}</p>;
}
