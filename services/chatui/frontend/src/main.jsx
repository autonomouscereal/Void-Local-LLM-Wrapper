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
      let raw = ''
      let data
      // Single fetch to neutral path; wait for full response
      const resp = await fetch('/api/call', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify({ conversation_id: conversationId, content: text })
      })
      raw = await resp.text()
      try { data = JSON.parse(raw) } catch { data = { error: raw || 'parse error' } }
      if (!(resp.status >= 200 && resp.status < 300)) {
        const errText = (raw || '').slice(0, 500) || `chat proxy error (${resp.status})`
        setMsgs(prev => ([...prev, { id: Date.now(), role: 'assistant', content: { text: `Error: ${errText}` } }]))
        return
      }
      const content = ((data.choices && data.choices[0] && data.choices[0].message) || {}).content || (data.error || raw)
      setMsgs(prev => ([...prev, { id: Date.now(), role: 'assistant', content: { text: content } }]))
      setText('')
    } finally {
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


