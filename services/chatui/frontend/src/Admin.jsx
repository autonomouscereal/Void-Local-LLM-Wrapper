import React, { useEffect, useMemo, useState } from "react";

async function jget(url){ const r=await fetch(url); if(!r.ok) throw new Error(await r.text()); return await r.json(); }
async function jpost(url, body){ const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}); if(!r.ok) throw new Error(await r.text()); return await r.json(); }
const fmtBytes = b => (b>=1<<30? (b/(1<<30)).toFixed(2)+" GB" : b>=1<<20? (b/(1<<20)).toFixed(2)+" MB" : b>=1<<10? (b/(1<<10)).toFixed(2)+" KB" : (b||0)+" B");

export default function AdminTab(){
  const [jobs,setJobs]=useState([]);
  const [replay,setReplay]=useState(null);
  const refreshJobs = async()=> setJobs((await jget("/jobs.list")).jobs || []);
  const replayJob = async(cid)=> setReplay((await jget(`/jobs.replay?id=${encodeURIComponent(cid)}`)).events || []);

  const [ttl,setTtl]=useState(30*24*3600);
  const [maxB,setMaxB]=useState(200*1024**3);
  const [gcPlan,setGcPlan]=useState(null);
  const [gcRes,setGcRes]=useState(null);
  const runDryGC = async()=> setGcPlan((await jpost("/artifacts.gc",{ttl_seconds:+ttl, max_bytes_total:+maxB, dryrun:true})).plan);
  const runGC    = async()=> setGcRes((await jpost("/artifacts.gc",{ttl_seconds:+ttl, max_bytes_total:+maxB, dryrun:false})).result);

  const [prompts,setPrompts]=useState([]);
  const [caps,setCaps]=useState(null);
  const loadPrompts = async()=> { try{ const r=await jget("/prompts/list"); setPrompts(r.prompts || r || []); }catch{ setPrompts([]); } };
  const loadCaps = async()=> setCaps(await jget("/capabilities.json"));

  const [dsName,setDsName]=useState("training");
  const [dsIncl,setDsIncl]=useState({facts:true, images:true, tts:true, music:true, research:true});
  const includeList = useMemo(()=>Object.entries(dsIncl).filter(([k,v])=>v).map(([k])=>k),[dsIncl]);
  const startExport = async()=>{ await jpost("/datasets/start",{ name: dsName, include: includeList }); alert("Export job started"); };

  useEffect(()=>{ refreshJobs(); loadCaps(); },[]);

  return (
    <div style={{display:"grid",gap:16}}>
      <section>
        <h3>Jobs</h3>
        <button onClick={refreshJobs} style={btn}>Refresh</button>
        <div style={{marginTop:8, display:"grid", gap:8}}>
          {(jobs||[]).map(j=> (
            <div key={j.id} style={card}>
              <div><b>{j.id}</b> — {j.tool}</div>
              <div>state: {j.state} | phase: {j.phase} | progress: {((j.progress||0)*100).toFixed(1)}%</div>
              <button style={btn} onClick={()=>replayJob(j.id)}>Replay (last 500)</button>
            </div>
          ))}
        </div>
        {replay && (<pre style={pre}>{JSON.stringify(replay,null,2)}</pre>)}
      </section>

      <section>
        <h3>GC / Retention</h3>
        <div style={row}>
          <label>TTL seconds <input value={ttl} onChange={e=>setTtl(e.target.value)} style={input}/></label>
          <label>Max bytes <input value={maxB} onChange={e=>setMaxB(e.target.value)} style={input}/></label>
          <button style={btn} onClick={runDryGC}>Dry-run</button>
          <button style={btnAccent} onClick={runGC}>Execute GC</button>
        </div>
        {gcPlan && (
          <div style={{marginTop:8}}>
            <div>Plan summary: total={fmtBytes(gcPlan.summary.bytes_total)} | TTL candidates={gcPlan.ttl} | Budget candidates={gcPlan.budget}</div>
          </div>
        )}
        {gcRes && (
          <div style={{marginTop:8}}>GC {gcRes.dryrun? "dry-run" : "executed"} — deleted: {(gcRes.deleted||[]).length}</div>
        )}
      </section>

      <section>
        <h3>Dataset Export</h3>
        <div style={row}>
          <label>Dataset name <input value={dsName} onChange={e=>setDsName(e.target.value)} style={input}/></label>
          {["facts","images","tts","music","research"].map(k=> (
            <label key={k} style={{display:"flex",alignItems:"center",gap:6}}>
              <input type="checkbox" checked={!!dsIncl[k]} onChange={e=>setDsIncl(s=>({...s,[k]:e.target.checked}))}/>
              {k}
            </label>
          ))}
          <button style={btn} onClick={startExport}>Start export</button>
        </div>
      </section>

      <section>
        <h3>Prompts</h3>
        <button style={btn} onClick={loadPrompts}>List</button>
        <pre style={pre}>{JSON.stringify(prompts,null,2)}</pre>
      </section>

      <section>
        <h3>Capabilities</h3>
        <button style={btn} onClick={loadCaps}>Reload</button>
        <pre style={pre}>{caps ? JSON.stringify(caps,null,2) : "—"}</pre>
      </section>
    </div>
  );
}

const card = { padding:8, border:"1px solid #334", borderRadius:8, background:"#12161e" };
const row  = { display:"flex", gap:8, flexWrap:"wrap", alignItems:"center" };
const btn  = { padding:"6px 10px", border:"1px solid #2a2f37", borderRadius:6, background:"#14181f", color:"#e6eef6", cursor:"pointer" };
const btnAccent = {...btn, border:"1px solid #3a8"};
const input= { width:160, background:"#14181f", color:"#e6eef6", border:"1px solid #2a2f37", borderRadius:6, padding:"6px 8px" };
const pre  = { whiteSpace:"pre-wrap", background:"#0e1116", padding:8, border:"1px solid #333", borderRadius:8, maxHeight:320, overflow:"auto", marginTop:8 };


