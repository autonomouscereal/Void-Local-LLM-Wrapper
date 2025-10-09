import React, { useEffect, useState, useRef } from 'react'
import { createRoot } from 'react-dom/client'
import { marked } from 'marked'

function App() {
  const [convos, setConvos] = useState([])
  const [cid, setCid] = useState(null)
  const [msgs, setMsgs] = useState([])
  const [text, setText] = useState('')
  const fileRef = useRef()
  const [jobs, setJobs] = useState([])
  const [showJobs, setShowJobs] = useState(true)
  const [sending, setSending] = useState(false)
  const localIdRef = useRef(1)
  const knownDoneJobsRef = useRef(new Set())
  const nextLocalId = () => {
    const n = localIdRef.current
    localIdRef.current = n + 1
    return n
  }

  const extractMedia = (text) => {
    const urls = []
    const re = /(https?:\/\/\S+)/g
    let m
    while ((m = re.exec(text)) !== null) {
      urls.push(m[1])
    }
    const uniq = Array.from(new Set(urls))
    const toMeta = (u) => {
      try {
        const o = new URL(u)
        const parts = o.pathname.split('/')
        const name = parts[parts.length - 1] || o.hostname
        return { url: u, name, host: o.hostname }
      } catch {
        return { url: u, name: u, host: '' }
      }
    }
    const images = uniq.filter(u => /\.(png|jpg|jpeg|gif|webp)$/i.test(u)).map(toMeta)
    const videos = uniq.filter(u => /\.(mp4|webm|mov|mkv)$/i.test(u)).map(toMeta)
    return { images, videos }
  }

  const renderChatContent = (text) => {
    const html = marked.parse(String(text || ''))
    const { images, videos } = extractMedia(text || '')
    return (
      <div>
        <div dangerouslySetInnerHTML={{ __html: html }} />
        {(images.length > 0 || videos.length > 0) && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, marginTop: 12 }}>
            {images.map((m, i) => (
              <div key={`img-${i}`} style={{ background: '#0b0b0f', border: '1px solid #222', borderRadius: 8, padding: 8 }}>
                <a href={m.url} target='_blank' rel='noreferrer'>
                  <img src={m.url} alt='' style={{ width: '100%', borderRadius: 6, display: 'block' }} />
                </a>
                <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 6, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.name}</div>
              </div>
            ))}
            {videos.map((m, i) => (
              <div key={`vid-${i}`} style={{ background: '#0b0b0f', border: '1px solid #222', borderRadius: 8, padding: 8 }}>
                <video src={m.url} controls style={{ width: '100%', borderRadius: 6, display: 'block' }} />
                <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 6, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.name}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }
  // Frontend request policy (history + rationale):
  // - Previous versions sometimes posted twice (to two endpoints) or used polling fallbacks. That led to
  //   confusing timing where the browser “errored” one request while another completed later, creating the
  //   impression of flakes. We now send exactly one awaited POST and await it fully in the UI.
  // - If you see a browser NetworkError yet the backend logs later show a 200, it generally means the client
  //   (or an intermediary) aborted/reset the connection. The backend now returns explicit Content-Length and
  //   Connection: close so the browser treats the response as definite and avoids chunked-transfer quirks.
  // - Parsing is based on content-type: JSON is parsed; otherwise we show raw text. Network failures are caught
  //   and surfaced as a single assistant “Error:” message instead of unhandled promise rejections.
  // - Historical attempts and current approach:
  //   * Tried fetch() and XMLHttpRequest back and forth: both implemented; XHR currently active for clarity.
  //   * Tried streaming keepalives from the proxy: removed after no improvement; now a single awaited POST.
  //   * Pointed UI to /api/chat (neutral path) to avoid any path-specific issues; still persisted errors reported.
  //   * CORS is unified server-side via a single global middleware stamping permissive headers on every response.

  async function refreshConvos() {
    const r = await fetch('/api/conversations')
    const j = await r.json()
    const list = j.data || []
    setConvos(list)
    return list
  }

  async function openConversation(id) {
    if (!id) return
    setCid(id)
    const r = await fetch(`/api/conversations/${id}/messages`)
    const j = await r.json()
    setMsgs(j.data || [])
  }

  async function newConversation() {
    const r = await fetch('/api/conversations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: 'Conversation' }) })
    const j = await r.json()
    await refreshConvos()
    await openConversation(j.id)
    return j.id
  }

  async function send() {
    if (!text.trim()) return
    let conversationId = cid
    if (!conversationId) {
      conversationId = await newConversation()
    }
    // Add user's message immediately to the UI
    const userText = text
    const userId = nextLocalId()
    setMsgs(prev => ([...prev, { id: userId, role: 'user', content: { text: userText } }]))
    setText('')
    setSending(true)
    try {
      // Persistent WebSocket: send chat message and await the server push
      const t0 = performance.now()
      console.log('[ui] ws connect')
      const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/api/ws')
      // Insert a temporary assistant placeholder to show thinking state
      const thinkingId = nextLocalId()
      const showThinking = () => {
        setMsgs(prev => ([...prev, { id: thinkingId, role: 'assistant', content: { text: 'Thinking…' } }]))
      }
      ws.onopen = () => {
        console.log('[ui] ws open')
        showThinking()
        ws.send(JSON.stringify({ conversation_id: conversationId, content: userText }))
      }
      ws.onmessage = (ev) => {
        const t1 = performance.now()
        const raw = ev.data || ''
        console.log('[ui] ws message', { bytes: raw.length, dt_ms: (t1 - t0).toFixed(1) })
        let data
        try { data = raw ? JSON.parse(raw) : {} } catch { data = { error: raw } }
        if (data && data.error) {
          // Replace thinking bubble with error
          setMsgs(prev => (prev.map(m => m.id === thinkingId ? { ...m, content: { text: `Error: ${data.error}` } } : m)))
          ws.close()
          setSending(false)
          return
        }
        // Prefer normalized message payload if provided; fallback to OpenAI-style choices
        let finalContent = ''
        if (data && data.message && data.message.role === 'assistant') {
          const mc = data.message.content
          if (typeof mc === 'string') finalContent = mc
          else if (mc && typeof mc.text === 'string') finalContent = mc.text
        }
        if (!finalContent) {
          const msgObj = (data && data.data && data.data.choices && data.data.choices[0] && data.data.choices[0].message) || {}
          finalContent = msgObj.content || data.text || data.error || raw
        }
        // Replace thinking bubble with final assistant content (never leave it empty)
        setMsgs(prev => (prev.map(m => m.id === thinkingId ? { ...m, content: { text: String(finalContent || '').trim() } } : m)))
        ws.close()
        setSending(false)
      }
      ws.onerror = () => {
        console.error('[ui] ws error')
        // Replace thinking bubble with error if present; else append error
        setMsgs(prev => {
          const hasThinking = prev.some(m => m.id === thinkingId)
          if (hasThinking) return prev.map(m => m.id === thinkingId ? { ...m, content: { text: 'Error: Network error' } } : m)
          return ([...prev, { id: Date.now(), role: 'assistant', content: { text: 'Error: Network error' } }])
        })
        setSending(false)
      }
      ws.onclose = () => {
        console.log('[ui] ws closed')
      }
      return
    } catch (err) {
      // Catch network-level errors so they don’t manifest as unhandled promise rejections.
      console.error('Network error while calling proxy', err)
      const msg = err && err.message ? err.message : 'Network error'
      setMsgs(prev => ([...prev, { id: Date.now(), role: 'assistant', content: { text: `Error: ${msg}` } }]))
    } finally {
      console.log('[ui] chat POST end')
      setSending(false)
    }
  }

  async function uploadFile(e) {
    if (!cid || !fileRef.current?.files?.length) return
    const f = fileRef.current.files[0]
    const fd = new FormData()
    fd.append('conversation_id', cid)
    fd.append('file', f)
    await fetch('/api/upload', { method: 'POST', body: fd })
    fileRef.current.value = ''
    alert('Uploaded! The orchestrator will see it in conversation context.')
  }

  useEffect(() => {
    (async () => {
      const list = await refreshConvos()
      if (list.length === 0) {
        await newConversation()
      } else {
        await openConversation(list[0].id)
      }
    })()
    // Jobs polling disabled by default; re-enable if needed
  }, [])

  // Auto-refresh jobs and surface completions with any discovered asset URLs into chat (enabled by default)
  useEffect(() => {
    let timer
    const extractUrls = (obj) => {
      const urls = []
      const walk = (v) => {
        if (!v) return
        if (typeof v === 'string') {
          if (/https?:\/\/\S+\.(mp4|mov|mkv|webm|png|jpg|jpeg|gif)/i.test(v)) urls.push(v)
        } else if (Array.isArray(v)) {
          v.forEach(walk)
        } else if (typeof v === 'object') {
          Object.values(v).forEach(walk)
        }
      }
      walk(obj)
      return Array.from(new Set(urls))
    }
    const tick = async () => {
      try {
        const r = await fetch('/api/jobs')
        const j = await r.json()
        const list = Array.isArray(j) ? j : (j.data || [])
        setJobs(list)
        for (const job of list) {
          const jid = job.id || job.job_id
          const status = (job.status || '').toLowerCase()
          if (jid && status === 'done' && !knownDoneJobsRef.current.has(jid)) {
            knownDoneJobsRef.current.add(jid)
            // Try to fetch job detail for URLs
            let detailText = ''
            try {
              const dr = await fetch(`/api/jobs/${encodeURIComponent(jid)}`)
              const dj = await dr.json()
              const urls = extractUrls(dj)
              if (urls.length) {
                detailText = `\nAssets:\n` + urls.map(u => `- ${u}`).join('\n')
              }
            } catch (_) {}
            setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: `Job ${jid} finished.${detailText}` } }]))
          }
        }
      } catch (_) {}
    }
    tick()
    timer = setInterval(tick, 4000)
    return () => { if (timer) clearInterval(timer) }
  }, [])

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'Inter, system-ui, Arial', background: '#0b0b0f', color: '#e6e6e6' }}>
      <div style={{ width: 320, borderRight: '1px solid #222', padding: 12 }}>
        <button onClick={newConversation} style={{ width: '100%', padding: 10, background: '#1f2937', color: '#fff', border: 'none', borderRadius: 6, marginBottom: 12 }}>New Conversation</button>
        <div style={{ overflowY: 'auto', maxHeight: 'calc(100vh - 80px)' }}>
          {(convos || []).map(c => (
            <div key={c.id} onClick={() => openConversation(c.id)} style={{ padding: 10, borderRadius: 6, marginBottom: 8, background: cid === c.id ? '#374151' : '#111827', cursor: 'pointer' }}>
              <div style={{ fontWeight: 600 }}>{c.title}</div>
              <div style={{ fontSize: 12, color: '#9ca3af' }}>{new Date(c.updated_at).toLocaleString()}</div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontWeight: 700 }}>Jobs</div>
            <label style={{ fontSize: 12, color: '#9ca3af' }}>auto-refresh</label>
          </div>
          <div style={{ marginTop: 8, overflowY: 'auto', maxHeight: 240, borderTop: '1px solid #222', paddingTop: 8 }}>
            {(jobs || []).map(j => (
              <div key={j.id || j.job_id} style={{ padding: 8, marginBottom: 6, background: '#0f172a', borderRadius: 6 }}>
                <div style={{ fontSize: 12, color: '#9ca3af' }}>{j.status || 'unknown'}</div>
                <div style={{ fontSize: 12 }}>job: {j.id || j.job_id}</div>
                {j.updated_at && <div style={{ fontSize: 11, color: '#6b7280' }}>{new Date(j.updated_at).toLocaleString()}</div>}
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="chat-pane" style={{ flex: 1 }}>
        <div className="chat-scroll" style={{ padding: 16 }}>
          {(msgs || []).map(m => (
            <div key={m.id} style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 12, color: '#9ca3af' }}>{m.role}</div>
              <div className="chat-bubble" style={{ padding: 12, background: m.role === 'assistant' ? '#111827' : '#0f172a', borderRadius: 8 }}>
                {renderChatContent(m.content?.text || '')}
              </div>
            </div>
          ))}
        </div>
        <div style={{ padding: 12, display: 'flex', gap: 8, borderTop: '1px solid #222', flexShrink: 0 }}>
          <input ref={fileRef} type='file' onChange={uploadFile} style={{ color: '#9ca3af' }} />
          <input value={text} onChange={e => setText(e.target.value)} placeholder='Type your prompt...' style={{ flex: 1, padding: 10, borderRadius: 6, border: '1px solid #333', background: '#0b0b0f', color: '#fff' }} />
          <button onClick={send} disabled={sending || !text.trim()} style={{ padding: '10px 16px', background: sending || !text.trim() ? '#16a34a' : '#22c55e', opacity: sending || !text.trim() ? 0.7 : 1, color: '#111', border: 'none', borderRadius: 6, fontWeight: 700, cursor: sending || !text.trim() ? 'not-allowed' : 'pointer' }}>{sending ? 'Sending…' : 'Send'}</button>
        </div>
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)


