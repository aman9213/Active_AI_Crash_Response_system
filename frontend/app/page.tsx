// "use client"
// import { useState } from "react"

// export default function Home() {
//   const [file, setFile] = useState<File | null>(null)
//   const [query, setQuery] = useState<string>("")
//   const [text, setText] = useState<string>("")
//   const [loading, setLoading] = useState<boolean>(false)
//   const [error, setError] = useState<string>("")

//   const send = async () => {
//     if (!file) return
//     setLoading(true)
//     setError("")
//     setText("")

//     try {
//       const fd = new FormData()
//       fd.append("file", file)
      
//       // Add optional query parameter
//       if (query.trim()) {
//         fd.append("query", query)
//       }

//       const res = await fetch("http://localhost:8000/caption", {
//         method: "POST",
//         body: fd,
//       })

//       if (!res.ok) {
//         throw new Error(`Server error: ${res.status} ${res.statusText}`)
//       }

//       const data = await res.json()
//       setText(data.description)
//     } catch (err: unknown) {
//       setError(err instanceof Error ? err.message : "Unknown error occurred")
//     } finally {
//       setLoading(false)
//     }
//   }

//   return (
//     <main style={{ padding: 40, maxWidth: 800, margin: "0 auto" }}>
//       <h2>🚗 AI Crash Response — Scene Analysis</h2>
//       <p style={{ color: "#666", marginBottom: 30 }}>
//         Upload a crash image or video and optionally ask specific questions about it
//       </p>

//       {/* File Upload */}
//       <div style={{ marginBottom: 20 }}>
//         <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>
//           Select Image or Video:
//         </label>
//         <input
//           type="file"
//           accept="image/*,video/*"
//           capture="environment"
//           onChange={e => setFile(e.target.files?.[0] ?? null)}
//           style={{
//             padding: 10,
//             border: "1px solid #ccc",
//             borderRadius: 4,
//             width: "100%",
//             boxSizing: "border-box",
//           }}
//         />
//       </div>

//       {/* Query Input */}
//       <div style={{ marginBottom: 20 }}>
//         <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>
//           Ask a Question (optional):
//         </label>
//         <textarea
//           value={query}
//           onChange={e => setQuery(e.target.value)}
//           placeholder="e.g., 'What vehicles are involved?', 'What is the damage level?', 'Are there injuries?'"
//           style={{
//             width: "100%",
//             padding: 10,
//             border: "1px solid #ccc",
//             borderRadius: 4,
//             fontFamily: "monospace",
//             minHeight: 80,
//             boxSizing: "border-box",
//             fontSize: 14,
//           }}
//         />
//         <p style={{ fontSize: 12, color: "#999", marginTop: 5 }}>
//           💡 Tip: Leave empty for a default analysis, or ask specific questions for targeted answers
//         </p>
//       </div>

//       {/* Analyze Button */}
//       <button
//         onClick={send}
//         disabled={!file || loading}
//         style={{
//           padding: "12px 24px",
//           fontSize: 16,
//           fontWeight: "bold",
//           backgroundColor: loading ? "#ccc" : "#007bff",
//           color: "white",
//           border: "none",
//           borderRadius: 4,
//           cursor: loading ? "default" : "pointer",
//           width: "100%",
//         }}
//       >
//         {loading ? "Analyzing…" : "Analyze Image/Video"}
//       </button>

//       {/* Error Display */}
//       {error && (
//         <p style={{ color: "red", marginTop: 16, fontWeight: "bold" }}>
//           ⚠️ Error: {error}
//         </p>
//       )}

//       {/* Results Display */}
//       {text && (
//         <div style={{ marginTop: 24 }}>
//           <h3 style={{ marginTop: 24, marginBottom: 12 }}>Analysis Result:</h3>
//           <pre
//             style={{
//               background: "#f4f4f4",
//               padding: 16,
//               borderRadius: 8,
//               whiteSpace: "pre-wrap",
//               wordWrap: "break-word",
//               fontSize: 14,
//               lineHeight: 1.6,
//               border: "1px solid #ddd",
//             }}
//           >
//             {text}
//           </pre>
//         </div>
//       )}
//     </main>
//   )
// }






"use client"
import { useState } from "react"

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [query, setQuery] = useState<string>("")
  const [text, setText] = useState<string>("")
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>("")
  const [model, setModel] = useState<string>("llava-1.5-q4")

  const models = [
    { value: "blip2-q8", label: "BLIP-2 Q8 (Fastest, smallest)" },
    { value: "llava-1.5-q4", label: "LLaVA-1.5 Q4 (Recommended)" },
    { value: "llava-1.5-q8", label: "LLaVA-1.5 Q8 (Better quality)" },
    { value: "llava-1.5", label: "LLaVA-1.5 Full (Best quality, slow)" },
    { value: "llava-next-video", label: "LLaVA-NeXT-Video (GPU only)" },
  ]

  const send = async () => {
    if (!file) return
    setLoading(true)
    setError("")
    setText("")

    try {
      const fd = new FormData()
      fd.append("file", file)
      fd.append("model", model)
      
      // Add optional query parameter
      if (query.trim()) {
        fd.append("query", query)
      }

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
    <main style={{ padding: 40, maxWidth: 800, margin: "0 auto" }}>
      <h2>🚗 AI Crash Response — Scene Analysis</h2>
      <p style={{ color: "#666", marginBottom: 30 }}>
        Upload a crash image or video and optionally ask specific questions about it
      </p>

      {/* Model Selection */}
      <div style={{ marginBottom: 20 }}>
        <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>
          Select AI Model:
        </label>
        <select
          value={model}
          onChange={e => setModel(e.target.value)}
          style={{
            width: "100%",
            padding: 10,
            border: "1px solid #ccc",
            borderRadius: 4,
            fontSize: 14,
            boxSizing: "border-box",
          }}
        >
          {models.map(m => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
        <p style={{ fontSize: 12, color: "#999", marginTop: 5 }}>
          💡 For Mac CPU: Use "LLaVA-1.5 Q4" (recommended balance of speed & quality)
        </p>
      </div>

      {/* File Upload */}
      <div style={{ marginBottom: 20 }}>
        <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>
          Select Image or Video:
        </label>
        <input
          type="file"
          accept="image/*,video/*"
          capture="environment"
          onChange={e => setFile(e.target.files?.[0] ?? null)}
          style={{
            padding: 10,
            border: "1px solid #ccc",
            borderRadius: 4,
            width: "100%",
            boxSizing: "border-box",
          }}
        />
      </div>

      {/* Query Input */}
      <div style={{ marginBottom: 20 }}>
        <label style={{ display: "block", marginBottom: 8, fontWeight: "bold" }}>
          Ask a Question (optional):
        </label>
        <textarea
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="e.g., 'What vehicles are involved?', 'What is the damage level?', 'Are there injuries?'"
          style={{
            width: "100%",
            padding: 10,
            border: "1px solid #ccc",
            borderRadius: 4,
            fontFamily: "monospace",
            minHeight: 80,
            boxSizing: "border-box",
            fontSize: 14,
          }}
        />
        <p style={{ fontSize: 12, color: "#999", marginTop: 5 }}>
          💡 Tip: Leave empty for a default analysis, or ask specific questions for targeted answers
        </p>
      </div>

      {/* Analyze Button */}
      <button
        onClick={send}
        disabled={!file || loading}
        style={{
          padding: "12px 24px",
          fontSize: 16,
          fontWeight: "bold",
          backgroundColor: loading ? "#ccc" : "#007bff",
          color: "white",
          border: "none",
          borderRadius: 4,
          cursor: loading ? "default" : "pointer",
          width: "100%",
        }}
      >
        {loading ? "Analyzing…" : "Analyze Image/Video"}
      </button>

      {/* Error Display */}
      {error && (
        <p style={{ color: "red", marginTop: 16, fontWeight: "bold" }}>
          ⚠️ Error: {error}
        </p>
      )}

      {/* Results Display */}
      {text && (
        <div style={{ marginTop: 24 }}>
          <h3 style={{ marginTop: 24, marginBottom: 12 }}>Analysis Result:</h3>
          <pre
            style={{
              background: "#f4f4f4",
              padding: 16,
              borderRadius: 8,
              whiteSpace: "pre-wrap",
              wordWrap: "break-word",
              fontSize: 14,
              lineHeight: 1.6,
              border: "1px solid #ddd",
            }}
          >
            {text}
          </pre>
        </div>
      )}
    </main>
  )
}