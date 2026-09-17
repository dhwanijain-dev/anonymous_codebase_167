"use client"

import { ChangeEvent, DragEvent, FormEvent, useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { RiAddLine, RiArrowUpLine, RiCheckboxCircleFill, RiCloseLine, RiDownloadLine, RiEarthLine, RiFileImageLine, RiLoader4Line, RiMore2Line, RiSparkling2Line, RiUploadCloud2Line, RiUser3Line } from "@remixicon/react"

type Box = [number, number, number, number]
type AnalysisResponse = { request_id: string; answer: string | null; tasks: Array<{ task: string; result: { answer: string; caption?: string | null; boxes?: Box[]; latency_seconds?: number }; model: Record<string, unknown> }>; execution_trace: { total_latency_seconds: number; selected_tasks: string[]; workflow: string[] } }
type Message = { role: "user" | "assistant"; text: string; files?: File[]; result?: AnalysisResponse }

const starterPrompts = ["Where are the vehicles in this image?", "How many buildings can you see?", "Describe the important features in this scene."]

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([])
  const [files, setFiles] = useState<File[]>([])
  const [prompt, setPrompt] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const fileInputRef = useRef<HTMLInputElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, busy])

  function addFiles(nextFiles: File[]) {
    const supported = nextFiles.filter((file) => /\.(tif|tiff|png|jpe?g)$/i.test(file.name))
    setFiles(supported.slice(0, 2))
    setError(supported.length === 0 ? "Use a GeoTIFF, TIFF, PNG, or JPEG image." : "")
  }
  function handleDrop(event: DragEvent<HTMLFormElement>) { event.preventDefault(); addFiles(Array.from(event.dataTransfer.files)) }
  function handleInput(event: ChangeEvent<HTMLInputElement>) { addFiles(Array.from(event.target.files ?? [])) }
  async function sendMessage(event?: FormEvent) {
    event?.preventDefault()
    if (busy || !prompt.trim()) return
    if (!files.length) return setError("Attach an image before sending your question.")
    const currentPrompt = prompt.trim(); const currentFiles = files
    setMessages((current) => [...current, { role: "user", text: currentPrompt, files: currentFiles }])
    setPrompt(""); setFiles([]); setError(""); setBusy(true)
    try {
      const form = new FormData(); form.append("query", currentPrompt); form.append("modalities", currentFiles.length === 2 ? "unknown,pair" : "unknown")
      currentFiles.forEach((file, index) => form.append(index === 0 ? "image1" : "image2", file, file.name))
      const response = await fetch("/api/backend/analyze", { method: "POST", body: form }); const body = await response.json()
      if (!response.ok) throw new Error(body.detail ?? "The analysis could not be completed.")
      setMessages((current) => [...current, { role: "assistant", text: body.answer ?? "No answer was returned.", result: body }])
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not reach the SatQuery backend.") } finally { setBusy(false) }
  }

  return <main className="chat-shell">
    <header className="chat-header"><div className="flex items-center gap-3"><div className="grid size-9 place-items-center rounded-xl bg-[#183d35] text-[#d8f566]"><RiEarthLine size={21} /></div><div><div className="font-heading text-lg font-semibold tracking-tight">SatQuery</div><div className="text-[10px] font-semibold tracking-[0.2em] text-[#83938a] uppercase">Earth intelligence</div></div></div><div className="flex items-center gap-3 text-xs text-[#718078]"><span className="hidden sm:inline">Private analysis session</span><span className="status-dot"><span /> Ready</span><Button variant="ghost" size="icon-sm" aria-label="More options"><RiMore2Line size={18} /></Button></div></header>
    <div className="chat-layout"><aside className="chat-sidebar"><Button variant="outline" className="w-full justify-start" onClick={() => { setMessages([]); setFiles([]); setError("") }}><RiAddLine size={16} /> New conversation</Button><div className="mt-8"><p className="eyebrow">What SatQuery can do</p><div className="mt-3 space-y-2 text-xs leading-5 text-[#718078]"><p>Answer questions about satellite imagery.</p><p>Find objects and mark their locations.</p><p>Compare image pairs automatically.</p></div></div><div className="mt-auto hidden border-t border-[#dce4dc] pt-4 lg:block"><p className="text-[10px] tracking-[0.16em] text-[#9aaa9f] uppercase">Supervisor routing</p><p className="mt-1 text-xs text-[#718078]">Automatic</p></div></aside>
      <section className="chat-main"><div className="chat-scroll">{messages.length === 0 ? <Welcome onPrompt={setPrompt} /> : messages.map((message, index) => <ChatMessage key={`${message.role}-${index}`} message={message} />)}{busy && <div className="message-row"><div className="avatar assistant-avatar"><RiSparkling2Line size={16} /></div><div className="message-content"><div className="message-name">SatQuery</div><div className="typing"><span /><span /><span /></div><p className="text-xs text-[#87968d]">Inspecting imagery and selecting evidence...</p></div></div>}<div ref={bottomRef} /></div>
        <form className="composer-wrap" onSubmit={sendMessage} onDrop={handleDrop} onDragOver={(event) => event.preventDefault()}><div className="composer"><div className="composer-files">{files.map((file) => <div key={file.name} className="file-chip"><RiFileImageLine size={14} /><span>{file.name}</span><button type="button" onClick={() => setFiles((current) => current.filter((item) => item !== file))} aria-label={`Remove ${file.name}`}><RiCloseLine size={14} /></button></div>)}</div><textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void sendMessage() } }} placeholder="Ask anything about your imagery..." rows={2} /><div className="flex items-center justify-between"><div className="flex items-center gap-1"><button type="button" className="composer-icon" onClick={() => fileInputRef.current?.click()} aria-label="Attach image"><RiUploadCloud2Line size={19} /></button><span className="text-[11px] text-[#91a098]">Drop up to 2 images here</span></div><Button type="submit" size="icon" disabled={busy || !prompt.trim() || !files.length} aria-label="Send message">{busy ? <RiLoader4Line className="animate-spin" size={17} /> : <RiArrowUpLine size={19} />}</Button></div></div><input ref={fileInputRef} type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" multiple onChange={handleInput} className="hidden" />{error && <p className="mt-2 rounded-lg border border-[#e4b8a9] bg-[#fff2ed] px-3 py-2 text-xs text-[#a04b3d]">{error}</p>}<p className="mt-2 text-center text-[10px] text-[#9aaa9f]">SatQuery can make mistakes. Check marked evidence against the image.</p></form>
      </section></div>
  </main>
}

function Welcome({ onPrompt }: { onPrompt: (prompt: string) => void }) { return <div className="welcome"><div className="grid size-14 place-items-center rounded-2xl bg-[#dcebd5] text-[#3e7754]"><RiSparkling2Line size={27} /></div><h1 className="mt-5 font-heading text-3xl font-semibold tracking-tight">What do you see?</h1><p className="mt-2 max-w-md text-center text-sm leading-6 text-[#718078]">Upload satellite imagery and ask a question in plain language. SatQuery will choose the right specialist and show its evidence.</p><div className="mt-7 grid max-w-2xl gap-2 sm:grid-cols-3">{starterPrompts.map((item) => <button key={item} onClick={() => onPrompt(item)} className="prompt-suggestion">{item}</button>)}</div></div> }

function ChatMessage({ message }: { message: Message }) {
  const grounding = message.result?.tasks.find((task) => task.task === "GROUNDING_CAPTIONING")?.result
  const boxes = grounding?.boxes ?? []
  return <div className={`message-row ${message.role === "user" ? "user-row" : ""}`}><div className={`avatar ${message.role === "user" ? "user-avatar" : "assistant-avatar"}`}>{message.role === "user" ? <RiUser3Line size={16} /> : <RiSparkling2Line size={16} />}</div><div className="message-content"><div className="message-name">{message.role === "user" ? "You" : "SatQuery"}</div>{message.files?.length ? <div className="message-images">{message.files.map((file) => <img key={file.name} src={URL.createObjectURL(file)} alt={file.name} />)}</div> : null}<div className={message.role === "user" ? "user-bubble" : "assistant-text"}>{message.text}</div>{grounding ? <div className="evidence-card"><div className="evidence-heading"><span><RiCheckboxCircleFill size={15} /> Grounding evidence</span><span>{boxes.length} region{boxes.length === 1 ? "" : "s"} marked</span></div><div className="evidence-image"><img src={message.files?.[0] ? URL.createObjectURL(message.files[0]) : ""} alt="Grounding evidence" />{boxes.map((box, index) => <div key={index} className="grounding-box" style={{ left: `${box[0] * 100}%`, top: `${box[1] * 100}%`, width: `${(box[2] - box[0]) * 100}%`, height: `${(box[3] - box[1]) * 100}%` }}><span>{index + 1}</span></div>)}</div>{grounding.caption && <div className="evidence-caption"><span className="eyebrow">Caption</span><p>{grounding.caption}</p></div>}<Button variant="outline" size="sm" className="mt-3" onClick={() => window.open(`data:application/json,${encodeURIComponent(JSON.stringify(message.result, null, 2))}`, "_blank")}><RiDownloadLine size={14} /> Export evidence</Button></div> : null}</div></div>
}
