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
  const [showJobs, setShowJobs] = useState(false)
  const [sending, setSending] = useState(false)
  // Frontend request policy (history + rationale):
  // - Previous versions sometimes posted twice (to two endpoints) or used polling fallbacks. That led to
  //   confusing timing where the browser “errored” one request while another completed later, creating the
  //   impression of flakes. We now send exactly one awaited POST and await it fully in the UI.
  // - If you see a browser NetworkError yet the backend logs later show a 200, it generally means the client
  //   (or an intermediary) aborted/reset the connection. The backend now returns explicit Content-Length and
  //   Connection: close so the browser treats the response as definite and avoids chunked-transfer quirks.
  // - Parsing is based on content-type: JSON is parsed; otherwise we show raw text. Network failures are caught
  //   and surfaced as a single assistant “Error:” message instead of unhandled promise rejections.

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
    setSending(true)
    try {
      // Single awaited POST to the proxy. No retries/polling here.
      const t0 = performance.now()
      console.log('[ui] chat POST start', { conversationId })
      const resp = await fetch(`/api/conversations/${conversationId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json, text/plain;q=0.9, */*;q=0.8' },
        body: JSON.stringify({ conversation_id: conversationId, content: text }),
        cache: 'no-store',
        mode: 'cors',
        credentials: 'omit',
        redirect: 'follow',
        referrerPolicy: 'no-referrer'
      })
      const t1 = performance.now()
      console.log('[ui] chat POST headers', { status: resp.status, ct: resp.headers.get('content-type') || '', dt_ms: (t1 - t0).toFixed(1) })
      const ct = resp.headers.get('content-type') || ''
      const raw = await resp.text()
      const t2 = performance.now()
      console.log('[ui] chat POST body', { bytes: raw ? raw.length : 0, dt_ms: (t2 - t0).toFixed(1) })
      let data
      if (ct.includes('application/json')) {
        try {
          data = raw ? JSON.parse(raw) : {}
        } catch (e) {
          console.warn('JSON parse failed, returning raw text', e)
          data = { error: raw }
        }
      } else {
        data = raw && raw.trim().length > 0 ? { text: raw } : { error: raw }
      }
      if (!resp.ok) {
        const errText = (raw || '').slice(0, 500) || `chat proxy error (${resp.status})`
        setMsgs(prev => ([...prev, { id: Date.now(), role: 'assistant', content: { text: `Error: ${errText}` } }]))
        return
      }
      const content = ((data.choices && data.choices[0] && data.choices[0].message) || {}).content || data.text || data.error || raw
      setMsgs(prev => ([...prev, { id: Date.now(), role: 'assistant', content: { text: content } }]))
      setText('')
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
              <div className="chat-bubble" style={{ padding: 12, background: m.role === 'assistant' ? '#111827' : '#0f172a', borderRadius: 8 }} dangerouslySetInnerHTML={{ __html: marked.parse((m.content?.text || '')) }} />
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


