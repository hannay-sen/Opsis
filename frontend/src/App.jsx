import { useRef, useState } from "react";

// In development the backend runs separately on port 8000.
// In the deployed build the backend serves this page, so requests go to the same site ("").
const API = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("idle"); // idle | loading | error
  const [error, setError] = useState("");
  const [tookMs, setTookMs] = useState(null);
  const [source, setSource] = useState(null); // what the current results are "like"
  const fileInput = useRef(null);

  async function run(request, describe) {
    setStatus("loading");
    setError("");
    try {
      const res = await request();
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setResults(data.results);
      setTookMs(data.took_ms);
      setSource(describe);
      setStatus("idle");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  }

  function searchText(e) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    run(() => fetch(`${API}/search/text?q=${encodeURIComponent(q)}`), { kind: "text", label: `“${q}”` });
  }

  function searchImage(file) {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    const preview = URL.createObjectURL(file);
    run(() => fetch(`${API}/search/image`, { method: "POST", body: form }), {
      kind: "image",
      label: "your photo",
      preview,
    });
  }

  function moreLikeThis(item) {
    run(() => fetch(`${API}/similar/${item.id}`), {
      kind: "item",
      label: [item.style, item.category].filter(Boolean).join(" ") || "this item",
      preview: `${API}/images/${item.path}`,
    });
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="logo">opsis<span>.</span></div>
        <form className="search" onSubmit={searchText}>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Describe a piece, e.g. walnut dining table"
            aria-label="Search by text"
          />
        </form>
        <button className="upload" onClick={() => fileInput.current?.click()}>
          Search by photo
        </button>
        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            searchImage(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </header>

      <main
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          searchImage(e.dataTransfer.files?.[0]);
        }}
      >
        {source && (
          <div className="context">
            {source.preview && <img src={source.preview} alt="" />}
            <p>
              More like <strong>{source.label}</strong>
              {tookMs != null && <span className="took"> · {results.length} results in {tookMs} ms</span>}
            </p>
          </div>
        )}

        {status === "error" && <p className="error">{error}</p>}

        {!source && status !== "loading" && (
          <div className="empty">
            <h1>Find pieces that feel like yours.</h1>
            <p>Describe a piece, upload a photo, or drop an image anywhere on the page.</p>
          </div>
        )}

        <div className={`grid ${status === "loading" ? "loading" : ""}`}>
          {results.map((item) => (
            <button key={item.id} className="card" onClick={() => moreLikeThis(item)} title="More like this">
              <img src={`${API}/images/${item.path}`} alt={`${item.style ?? ""} ${item.category ?? ""}`} loading="lazy" />
              <div className="meta">
                <span>{[item.style, item.category].filter(Boolean).join(" · ")}</span>
                <span className="score">{(item.score * 100).toFixed(0)}%</span>
              </div>
            </button>
          ))}
        </div>
      </main>
    </div>
  );
}