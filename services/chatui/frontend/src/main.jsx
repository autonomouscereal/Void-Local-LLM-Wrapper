import React, { useEffect, useState, useRef } from 'react'
import { createRoot } from 'react-dom/client'
import { marked } from 'marked'

// UI notes:
// - Single-send semantics (exactly one request per Send) with a "Thinking…" placeholder.
// - Assistant replies are rendered as Markdown; we embed images/videos and show (no content) if empty.
// - Film/job progress: job_id(s) are parsed from assistant messages; we auto-subscribe to
//   /api/jobs/{id}/stream (SSE) and update inline progress bubbles until succeeded/failed,
//   then embed discovered asset URLs.
// - We keep the full assistant response and only append status/IDs; nothing is overwritten.

function App() {
  const [convos, setConvos] = useState([])
  const [cid, setCid] = useState(null)
  const [msgs, setMsgs] = useState([])
  const [text, setText] = useState('')
  const fileRef = useRef()
  const [jobs, setJobs] = useState([])
  const [showJobs, setShowJobs] = useState(true)
  const [sending, setSending] = useState(false)
  const [voiceOn, setVoiceOn] = useState(false)
  const recognizerRef = useRef(null)
  const [makeMode, setMakeMode] = useState('auto')
  const [activeTab, setActiveTab] = useState('chat')
  // Image panel refs
  const imgModeRef = useRef()
  const imgSizeRef = useRef()
  const imgPromptRef = useRef()
  const imgSeedRef = useRef()
  const imgRefIdRef = useRef()
  const imgFileRef = useRef()
  // TTS panel refs
  const ttsTextRef = useRef()
  const ttsVoiceRef = useRef()
  const ttsVoiceIdRef = useRef()
  const ttsRateRef = useRef()
  const ttsPitchRef = useRef()
  const ttsSeedRef = useRef()
  // Music panel refs
  const musicPromptRef = useRef()
  const musicBpmRef = useRef()
  const musicLenRef = useRef()
  const musicRefIdRef = useRef()
  const musicSeedRef = useRef()
  const localIdRef = useRef(1)
  const knownDoneJobsRef = useRef(new Set())
  const jobsTimerRef = useRef(null)
  const progressStreamsRef = useRef({})
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
    const audios = uniq.filter(u => /\.(wav|mp3|m4a|ogg|flac|opus)$/i.test(u)).map(toMeta)
    return { images, videos, audios }
  }

  const renderChatContent = (text) => {
    const raw = String(text || '')
    const html = marked.parse(raw)
    const { images, videos, audios } = extractMedia(text || '')
    const hasText = raw.trim().length > 0
    const htmlLooksEmpty = (String(html || '').replace(/<[^>]*>/g, '').trim().length === 0)
    return (
      <div>
        {hasText ? (
          htmlLooksEmpty ? (
            <pre style={{ whiteSpace: 'pre-wrap', fontSize: 13, color: '#e5e7eb', background: 'transparent', margin: 0 }}>{raw}</pre>
          ) : (
            <div dangerouslySetInnerHTML={{ __html: html }} />
          )
        ) : (
          (images.length === 0 && videos.length === 0) ? (
            <div style={{ fontSize: 12, color: '#9ca3af', fontStyle: 'italic' }}>(no content)</div>
          ) : null
        )}
        {(images.length > 0 || videos.length > 0 || audios.length > 0) && (
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
            {audios.map((m, i) => (
              <div key={`aud-${i}`} style={{ background: '#0b0b0f', border: '1px solid #222', borderRadius: 8, padding: 8 }}>
                <audio src={m.url} controls style={{ width: '100%', display: 'block' }} />
                <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 6, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.name}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  // ---- Live Job Progress (SSE) ----
  const parseJobIdsFromText = (txt) => {
    const ids = new Set()
    if (!txt) return []
    // 1) Look for a "jobs:" line with backticked IDs
    const jobLineMatch = txt.match(/\bjobs:\s*([^\n]+)/i)
    if (jobLineMatch) {
      const line = jobLineMatch[1]
      const backticked = Array.from(line.matchAll(/`([^`]+)`/g)).map(m => m[1])
      backticked.forEach(x => ids.add(x.trim()))
    }
    // 2) Generic job id patterns (UUID-like or hex strings)
    const generic = Array.from(txt.matchAll(/\bjob[_\s-]*id\b[:\s]*([a-z0-9-]{8,})/ig)).map(m => m[1])
    generic.forEach(x => ids.add(x.trim()))
    return Array.from(ids)
  }

  const startJobStream = (jobId) => {
    if (!jobId || progressStreamsRef.current[jobId]) return
    // Create a placeholder assistant bubble with indeterminate progress
    const placeholderId = nextLocalId()
    setMsgs(prev => ([...prev, { id: placeholderId, role: 'assistant', content: { text: `Job ${jobId} queued…` } }]))
    try {
      const src = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/stream?interval_ms=1000`)
      progressStreamsRef.current[jobId] = { src, msgId: placeholderId }
      src.onmessage = (ev) => {
        if (!ev?.data) return
        if (ev.data === '[DONE]') {
          try { src.close() } catch {}
          delete progressStreamsRef.current[jobId]
          knownDoneJobsRef.current.add(jobId)
          return
        }
        let snapshot = {}
        try { snapshot = JSON.parse(ev.data) } catch { /* ignore */ }
        const status = (snapshot.status || '').toLowerCase()
        const assets = snapshot.result || {}
        let detailText = ''
        // Try to extract asset URLs from result
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
        walk(assets)
        if (urls.length) {
          detailText = `\nAssets:\n` + Array.from(new Set(urls)).map(u => `- ${u}`).join('\n')
        }
        const label = status || 'running'
        setMsgs(prev => prev.map(m => m.id === placeholderId ? { ...m, content: { text: `Job ${jobId} ${label}.${detailText}` } } : m))
      }
      src.onerror = () => {
        try { src.close() } catch {}
        delete progressStreamsRef.current[jobId]
      }
    } catch {
      // ignore stream failures; polling fallback still exists
    }
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
    // Quick per-modality path for non-power users
    if (makeMode !== 'auto') {
      try {
        if (makeMode === 'image') {
          const resp = await callTool('image.dispatch', { mode: 'gen', prompt: text.trim(), size: '1024x1024' })
          const cid = resp?.result?.meta?.cid
          const id = resp?.result?.tool_calls?.[0]?.result_ref
          const url = (cid && id) ? `/uploads/artifacts/image/${cid}/${id}` : ''
          setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: url || JSON.stringify(resp) } }]))
          setText('')
          return
        }
        if (makeMode === 'tts') {
          const resp = await callTool('tts.speak', { text: text.trim() })
          const cid = resp?.result?.meta?.cid
          const id = resp?.result?.tool_calls?.[0]?.result_ref
          const url = (cid && id) ? `/uploads/artifacts/audio/tts/${cid}/${id}` : ''
          setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: url || JSON.stringify(resp) } }]))
          setText('')
          return
        }
        if (makeMode === 'music') {
          const resp = await callTool('music.compose', { prompt: text.trim(), length_s: 30 })
          const cid = resp?.result?.meta?.cid
          const id = resp?.result?.tool_calls?.[0]?.result_ref
          const url = (cid && id) ? `/uploads/artifacts/music/${cid}/${id}` : ''
          setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: url || JSON.stringify(resp) } }]))
          setText('')
          return
        }
      } catch (e) {
        setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: `Error: ${e?.message || e}` } }]))
        return
      }
    }
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
        const finalText = String(finalContent || '').trim()
        setMsgs(prev => (prev.map(m => m.id === thinkingId ? { ...m, content: { text: finalText } } : m)))
        // If voice mode is on, speak the assistant reply via Web Speech
        if (voiceOn && finalText) {
          try {
            const u = new SpeechSynthesisUtterance(finalText)
            u.rate = 1.0; u.pitch = 1.0
            window.speechSynthesis.cancel()
            window.speechSynthesis.speak(u)
          } catch {}
        }
        // Scan for job IDs and start live progress streams
        const jobIds = parseJobIdsFromText(finalText)
        jobIds.forEach(startJobStream)
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

  // --- Voice Chat (Browser SpeechRecognition + SpeechSynthesis) ---
  const toggleVoice = async () => {
    const next = !voiceOn
    setVoiceOn(next)
    if (!next) {
      try { recognizerRef.current && recognizerRef.current.stop && recognizerRef.current.stop() } catch {}
      return
    }
    try {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition
      if (!SR) {
        alert('SpeechRecognition API not available in this browser')
        setVoiceOn(false)
        return
      }
      const rec = new SR()
      rec.lang = 'en-US'
      rec.continuous = false
      rec.interimResults = false
      rec.maxAlternatives = 1
      rec.onresult = async (ev) => {
        try {
          const txt = (ev.results && ev.results[0] && ev.results[0][0] && ev.results[0][0].transcript) || ''
          if (txt && txt.trim()) {
            setText(txt)
            await send()
          }
        } catch {}
      }
      rec.onend = () => {
        if (voiceOn) {
          try { rec.start() } catch {}
        }
      }
      recognizerRef.current = rec
      rec.start()
    } catch (e) {
      console.error('voice error', e)
      setVoiceOn(false)
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

  // ------- Step 20: Tool runner helpers -------
  async function uploadOne(file) {
    const fd = new FormData()
    fd.append('file', file)
    const r = await fetch('/api/upload', { method: 'POST', body: fd })
    const j = await r.json()
    // Prefer URL if provided, else construct uploads path
    if (j && typeof j.url === 'string' && j.url.includes('/uploads/')) return j.url
    if (j && typeof j.name === 'string') return '/uploads/' + j.name
    return ''
  }
  async function uploadMany(fileList) {
    const out = []
    for (const f of fileList) {
      const p = await uploadOne(f)
      if (p) out.push(p)
    }
    return out
  }
  async function callTool(name, args) {
    const r = await fetch('/api/tool.run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, args }) })
    const j = await r.json()
    return j
  }
  async function runImage() {
    const mode = imgModeRef.current?.value || 'gen'
    const size = imgSizeRef.current?.value || ''
    const prompt = imgPromptRef.current?.value || ''
    const seed = imgSeedRef.current?.value ? Number(imgSeedRef.current.value) : null
    const refid = imgRefIdRef.current?.value || ''
    const file = imgFileRef.current?.files?.[0]
    const args = { mode, prompt, size, seed, ref_ids: refid ? [refid] : [] }
    if (mode !== 'gen' && file) {
      const url = await uploadOne(file)
      // Map to path if needed
      args.image_ref = url
    }
    const resp = await callTool('image.dispatch', args)
    if (resp && resp.result && resp.result.meta && resp.result.tool_calls) {
      const cid = resp.result.meta.cid
      const tc = resp.result.tool_calls[0] || {}
      const resultRef = tc.result_ref
      if (cid && resultRef) {
        const url = `/uploads/artifacts/image/${cid}/${resultRef}`
        setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: `Generated: ${url}` } }]))
      } else {
        setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: JSON.stringify(resp.result) } }]))
      }
    } else {
      setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: JSON.stringify(resp) } }]))
    }
  }
  async function runTTS() {
    const textV = ttsTextRef.current?.value || ''
    const voice = ttsVoiceRef.current?.value || ''
    const voice_id = ttsVoiceIdRef.current?.value || ''
    const rate = ttsRateRef.current?.value ? Number(ttsRateRef.current.value) : 1.0
    const pitch = ttsPitchRef.current?.value ? Number(ttsPitchRef.current.value) : 0.0
    const seed = ttsSeedRef.current?.value ? Number(ttsSeedRef.current.value) : null
    const args = { text: textV, voice, voice_id, rate, pitch, seed }
    const resp = await callTool('tts.speak', args)
    if (resp && resp.result && resp.result.meta && resp.result.tool_calls) {
      const cid = resp.result.meta.cid
      const tc = resp.result.tool_calls[0] || {}
      const resultRef = tc.result_ref
      if (cid && resultRef) {
        const url = `/uploads/artifacts/audio/tts/${cid}/${resultRef}`
        setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: `Spoken: ${url}` } }]))
      } else {
        setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: JSON.stringify(resp.result) } }]))
      }
    } else {
      setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: JSON.stringify(resp) } }]))
    }
  }
  async function runMusic() {
    const prompt = musicPromptRef.current?.value || ''
    const bpm = musicBpmRef.current?.value ? Number(musicBpmRef.current.value) : null
    const length_s = musicLenRef.current?.value ? Number(musicLenRef.current.value) : null
    const music_id = musicRefIdRef.current?.value || ''
    const seed = musicSeedRef.current?.value ? Number(musicSeedRef.current.value) : null
    const args = { prompt, bpm, length_s, music_id: music_id || null, seed }
    const resp = await callTool('music.compose', args)
    if (resp && resp.result && resp.result.meta && resp.result.tool_calls) {
      const cid = resp.result.meta.cid
      const tc = resp.result.tool_calls[0] || {}
      const resultRef = tc.result_ref
      if (cid && resultRef) {
        const url = `/uploads/artifacts/music/${cid}/${resultRef}`
        setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: `Track: ${url}` } }]))
      } else {
        setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: JSON.stringify(resp.result) } }]))
      }
    } else {
      setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: JSON.stringify(resp) } }]))
    }
  }
  async function saveRef() {
    const kindEl = document.getElementById('ref-kind')
    const titleEl = document.getElementById('ref-title')
    const embEl = document.getElementById('ref-emb')
    const filesEl = document.getElementById('ref-files')
    const kind = kindEl.value
    const title = titleEl.value
    const compute_embeds = embEl.checked
    const files = filesEl.files
    let payload = { kind, title, files: {}, compute_embeds }
    if (kind === 'image') {
      const paths = await uploadMany(files)
      payload.files.images = paths
    } else if (kind === 'voice') {
      const paths = await uploadMany(files)
      payload.files.voice_samples = paths
    } else if (kind === 'music') {
      const paths = await uploadMany(files)
      if (paths.length > 0) payload.files.track = paths[0]
      if (paths.length > 1) payload.files.stems = paths.slice(1)
    }
    const r = await fetch('/api/refs.save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
    const j = await r.json()
    const out = document.getElementById('ref-list')
    out.textContent = JSON.stringify(j, null, 2)
  }
  async function listRefs() {
    const kindEl = document.getElementById('ref-kind')
    const kind = kindEl.value
    const r = await fetch('/api/refs.list' + (kind ? ('?kind=' + encodeURIComponent(kind)) : ''))
    const j = await r.json()
    const out = document.getElementById('ref-list')
    out.textContent = JSON.stringify(j, null, 2)
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
          // Consider orchestrator statuses; treat succeeded/failed as terminal (done)
          const isDone = status === 'done' || status === 'succeeded' || status === 'failed'
          if (jid && isDone && !knownDoneJobsRef.current.has(jid)) {
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
            const statusLabel = status === 'failed' ? 'failed' : 'finished'
            setMsgs(prev => ([...prev, { id: nextLocalId(), role: 'assistant', content: { text: `Job ${jid} ${statusLabel}.${detailText}` } }]))
          }
        }
      } catch (_) {}
    }
    // Prevent duplicate intervals (e.g., React StrictMode dev double-mount)
    if (jobsTimerRef.current) {
      clearInterval(jobsTimerRef.current)
      jobsTimerRef.current = null
    }
    tick()
    jobsTimerRef.current = setInterval(tick, 5000)
    return () => { if (jobsTimerRef.current) { clearInterval(jobsTimerRef.current); jobsTimerRef.current = null } }
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
        {/* Tabs */}
        <div style={{ display: 'flex', gap: 8, padding: 12, borderBottom: '1px solid #222' }}>
          {['chat','images','tts','music','refs'].map(t => (
            <button key={t} onClick={() => setActiveTab(t)} style={{ padding: '6px 10px', background: activeTab===t ? '#1f2937' : '#0f172a', border: '1px solid #333', borderRadius: 6, color: '#e6e6e6' }}>{t.toUpperCase()}</button>
          ))}
        </div>

        {/* Panels */}
        {activeTab === 'chat' && (
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
        )}

        {activeTab === 'images' && (
          <div style={{ padding: 16 }}>
            <div className="row" style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
              <select ref={imgModeRef} defaultValue="gen"><option value="gen">gen</option><option value="edit">edit</option><option value="upscale">upscale</option></select>
              <input ref={imgSizeRef} placeholder="size e.g. 1024x1024" />
              <input ref={imgRefIdRef} placeholder="ref_id (optional)" />
              <input ref={imgSeedRef} placeholder="seed (optional)" />
            </div>
            <textarea ref={imgPromptRef} rows={3} placeholder="Describe the image…" style={{ width: '100%', marginBottom: 8 }} />
            <div className="row" style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
              <input type='file' ref={imgFileRef} />
              <button onClick={runImage}>Run</button>
            </div>
          </div>
        )}

        {activeTab === 'tts' && (
          <div style={{ padding: 16 }}>
            <textarea ref={ttsTextRef} rows={3} placeholder="Text to speak…" style={{ width: '100%', marginBottom: 8 }} />
            <div className="row" style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
              <input ref={ttsVoiceRef} placeholder="voice preset/name" />
              <input ref={ttsVoiceIdRef} placeholder="voice ref_id (optional)" />
              <input ref={ttsRateRef} placeholder="rate (1.0)" defaultValue="1.0" />
              <input ref={ttsPitchRef} placeholder="pitch (0.0)" defaultValue="0.0" />
              <input ref={ttsSeedRef} placeholder="seed (optional)" />
            </div>
            <button onClick={runTTS}>Speak</button>
          </div>
        )}

        {activeTab === 'music' && (
          <div style={{ padding: 16 }}>
            <textarea ref={musicPromptRef} rows={3} placeholder="Music prompt (mood/genre)…" style={{ width: '100%', marginBottom: 8 }} />
            <div className="row" style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
              <input ref={musicBpmRef} placeholder="bpm" />
              <input ref={musicLenRef} placeholder="length_s" />
              <input ref={musicRefIdRef} placeholder="music ref_id (optional)" />
              <input ref={musicSeedRef} placeholder="seed (optional)" />
            </div>
            <button onClick={runMusic}>Compose</button>
          </div>
        )}

        {activeTab === 'refs' && (
          <div style={{ padding: 16 }}>
            <div className="row" style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
              <select id='ref-kind' defaultValue='image'><option value='image'>image</option><option value='voice'>voice</option><option value='music'>music</option></select>
              <input id='ref-title' placeholder='title' />
              <label className='small' style={{ display:'flex', alignItems:'center', gap:6 }}>compute embeds <input type='checkbox' id='ref-emb' defaultChecked /></label>
            </div>
            <div className='row' style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
              <input type='file' id='ref-files' multiple />
              <button onClick={saveRef}>Save Ref</button>
              <button onClick={listRefs}>List</button>
            </div>
            <pre id='ref-list' className='log' style={{ whiteSpace:'pre-wrap' }}></pre>
          </div>
        )}

        {/* Chat input stays always visible */}
        <div style={{ padding: 12, display: 'flex', gap: 8, borderTop: '1px solid #222', flexShrink: 0 }}>
          <input ref={fileRef} type='file' onChange={uploadFile} style={{ color: '#9ca3af' }} />
          <input value={text} onChange={e => setText(e.target.value)} placeholder='Type your prompt...' style={{ flex: 1, padding: 10, borderRadius: 6, border: '1px solid #333', background: '#0b0b0f', color: '#fff' }} />
          <select value={makeMode} onChange={e => setMakeMode(e.target.value)} style={{ padding: '10px 8px' }}>
            <option value='auto'>Auto</option>
            <option value='image'>Image</option>
            <option value='tts'>TTS</option>
            <option value='music'>Music</option>
          </select>
          <button onClick={toggleVoice} style={{ padding: '10px 12px', background: voiceOn ? '#f59e0b' : '#374151', color: '#fff', border: 'none', borderRadius: 6 }}>{voiceOn ? 'Voice: ON' : 'Voice: OFF'}</button>
          <button onClick={send} disabled={sending || !text.trim()} style={{ padding: '10px 16px', background: sending || !text.trim() ? '#16a34a' : '#22c55e', opacity: sending || !text.trim() ? 0.7 : 1, color: '#111', border: 'none', borderRadius: 6, fontWeight: 700, cursor: sending || !text.trim() ? 'not-allowed' : 'pointer' }}>{sending ? 'Sending…' : 'Send'}</button>
        </div>
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)


