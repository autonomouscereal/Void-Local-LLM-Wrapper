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

  async function refreshConvos() {
    const r = await fetch('/api/conversations')
    const j = await r.json()
    const list = j.data || []
    setConvos(list)
    return list
  }

  async function openConversation(id) {
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
    try {
      const r = await fetch(`/api/conversations/${conversationId}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content: text }) })
      const j = await r.json()
      setText('')
      await openConversation(conversationId)
    } catch (e) {
      console.error('send failed', e)
      alert('Send failed: ' + (e?.message || e))
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
    const iv = setInterval(async () => {
      try {
        const r = await fetch('/api/jobs')
        const j = await r.json()
        setJobs(j.data || [])
      } catch {}
    }, 3000)
    return () => clearInterval(iv)
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
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <div style={{ flex: 1, padding: 16, overflowY: 'auto' }}>
          {(msgs || []).map(m => (
            <div key={m.id} style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 12, color: '#9ca3af' }}>{m.role}</div>
              <div style={{ padding: 12, background: m.role === 'assistant' ? '#111827' : '#0f172a', borderRadius: 8 }} dangerouslySetInnerHTML={{ __html: marked.parse((m.content?.text || '')) }} />
            </div>
          ))}
        </div>
        <div style={{ padding: 12, display: 'flex', gap: 8, borderTop: '1px solid #222' }}>
          <input ref={fileRef} type='file' onChange={uploadFile} style={{ color: '#9ca3af' }} />
          <input value={text} onChange={e => setText(e.target.value)} placeholder='Type your prompt...' style={{ flex: 1, padding: 10, borderRadius: 6, border: '1px solid #333', background: '#0b0b0f', color: '#fff' }} />
          <button onClick={send} style={{ padding: '10px 16px', background: '#22c55e', color: '#111', border: 'none', borderRadius: 6, fontWeight: 700 }}>Send</button>
        </div>
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)


