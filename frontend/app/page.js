"use client"
import { useState } from "react"

export default function Home() {

  const [file,setFile]=useState(null)
  const [text,setText]=useState("")

  const send = async () => {

    const fd=new FormData()
    fd.append("file",file)

    const res=await fetch("http://localhost:8000/caption",{
      method:"POST",
      body:fd
    })

    const data=await res.json()
    setText(data.description)
  }

  return (
    <main style={{padding:40}}>

      <h2>LLaVA-NeXT Video Caption</h2>

      <input
        type="file"
        accept="image/*,video/*"
        capture="environment"
        onChange={e=>setFile(e.target.files[0])}
      />

      <br/><br/>

      <button onClick={send}>Generate</button>

      <pre>{text}</pre>

    </main>
  )
}