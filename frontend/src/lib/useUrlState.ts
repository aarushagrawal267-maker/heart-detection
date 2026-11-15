import { useSearchParams } from "react-router-dom";

/** Keep simple page state (record, beat, …) in the URL so views are shareable/bookmarkable. */
export function useUrlState(key: string, fallback: string): [string, (v: string) => void] {
  const [params, setParams] = useSearchParams();
  const value = params.get(key) ?? fallback;
  const set = (v: string) => {
    const next = new URLSearchParams(params);
    if (v === fallback) next.delete(key);
    else next.set(key, v);
    setParams(next, { replace: true });
  };
  return [value, set];
}
