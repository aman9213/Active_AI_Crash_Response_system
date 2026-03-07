"use client"
import { useState } from "react"

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [text, setText] = useState<string>("")
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>("")

  const send = async () => {
    if (!file) return
    setLoading(true)
    setError("")
    setText("")

    try {
      const fd = new FormData()
      fd.append("file", file)

      const res = await fetch("http://localhost:8000/caption", {
        method: "POST",
        body: fd,
      })

      if (!res.ok) {
        throw new Error(`Server error: ${res.status} ${res.statusText}`)
      }

      const data = await res.json()
      setText(data.description)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error occurred")
    } finally {
      setLoading(false)
    }
  }

  return (
    <main style={{ padding: 40, maxWidth: 720, margin: "0 auto" }}>
      <h2>🚗 AI Crash Response — Scene Caption</h2>

      <input
        type="file"
        accept="image/*,video/*"
        capture="environment"
        onChange={e => setFile(e.target.files?.[0] ?? null)}
      />

      <br /><br />

      <button onClick={send} disabled={!file || loading}>
        {loading ? "Generating…" : "Generate Caption"}
      </button>

      {error && (
        <p style={{ color: "red", marginTop: 16 }}>⚠️ {error}</p>
      )}

      {text && (
        <pre style={{ marginTop: 24, whiteSpace: "pre-wrap", background: "#f4f4f4", padding: 16, borderRadius: 8 }}>
          {text}
        </pre>
      )}
    </main>
  )
}
