"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { clockTime, dragTarget, draftStorageKey, frameAt, makeAdjustment, markingName, markingPath, nearestMarking, parseDraft, projectMarkings, type Adjustment, type AlignmentClip, type AlignmentPacket, type AlignmentState, type MarkingHit, type Point } from "@/lib/alignment";
import styles from "./alignment.module.css";

/* Operate: extend Night Pitch with direct manipulation of source-bound markings.
   A desktop owner pauses a real clip, drags the visible paint, checks the pan,
   and saves. The video owns the space; controls stay beside it. Amber marks an
   approximate alignment; confirmation records a visual review only. */
async function getJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {cache:"no-store",...init});
  const data = await response.json() as T & {error?:string};
  if (response.status === 409) throw new Error("This clip was saved in another tab. Reload its saved alignment before saving again.");
  if (!response.ok || data.error) throw new Error(data.error || "The clip could not be loaded. Try again.");
  return data;
}

type Session = { packet: AlignmentPacket; state: AlignmentState };
export default function AlignmentWorkspace({clips}: {clips: AlignmentClip[]}) {
  const [clipId,setClipId] = useState(clips.find(c=>c.ready)?.id ?? clips[0]?.id ?? "");
  const [dirty,setDirty] = useState(false);
  return <section className={styles.workspace} aria-label="Field alignment">
    <header className={styles.heading}>
      <h1 className={styles.srOnly}>Align the field</h1>
      <label className={styles.clipPicker}>Review clip<select value={clipId} disabled={dirty} onChange={e=>setClipId(e.target.value)}>
        {clips.map(clip=><option key={clip.id} value={clip.id} disabled={!clip.ready}>{clip.title} · {clip.description}{clip.ready ? "" : " · preparing"}</option>)}
      </select></label>
    </header>
    {clips.some(c=>c.id===clipId && c.ready) ? <ClipLoader key={clipId} id={clipId} onDirty={setDirty}/> : <div className={styles.empty}><h2>No review clip is ready yet</h2><p>Prepared footage will appear here once its field mapping is available. Refresh this page after preparation finishes.</p><button onClick={()=>window.location.reload()}>Refresh clips</button></div>}
  </section>;
}
function ClipLoader({id,onDirty}: {id:string;onDirty:(dirty:boolean)=>void}) {
  const [session,setSession] = useState<Session|null>(null);
  const [error,setError] = useState("");
  const [retry,setRetry] = useState(0);
  useEffect(()=>{
    const controller = new AbortController();
    Promise.all([
      getJSON<AlignmentPacket>(`/api/alignment/${id}/packet`,{signal:controller.signal}),
      getJSON<AlignmentState>(`/api/alignment/${id}/state`,{signal:controller.signal}),
    ]).then(([packet,state])=>setSession({packet,state})).catch((e:Error)=>{if(!controller.signal.aborted)setError(e.message);});
    return ()=>controller.abort();
  },[id,retry]);
  if(error) return <div className={styles.empty} role="alert"><h2>This clip could not be opened</h2><p>{error}</p><button onClick={()=>{setError("");setRetry(n=>n+1);}}>Try again</button></div>;
  if(!session) return <div className={styles.empty} role="status"><h2>Opening the review clip…</h2><p>Loading the footage and its saved field alignment.</p></div>;
  return <Editor packet={session.packet} initial={session.state} onDirty={onDirty}/>;
}

type Hit = MarkingHit;
type Drag = {hit:Hit;pointer:Point;frameIndex:number;mode:Adjustment["mode"];operation:Adjustment|null};
function Editor({packet,initial,onDirty}: {packet:AlignmentPacket;initial:AlignmentState;onDirty:(dirty:boolean)=>void}) {
  const video = useRef<HTMLVideoElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<Drag|null>(null);
  const storageKey=draftStorageKey(packet);
  const [recovered] = useState(()=>{try{return parseDraft(sessionStorage.getItem(storageKey),initial);}catch{return null;}});
  const [saved,setSaved] = useState(initial);
  const [history,setHistory] = useState<Adjustment[][]>([recovered?.operations ?? initial.operations]);
  const [position,setPosition] = useState(0);
  const [draft,setDraft] = useState<Adjustment|null>(null);
  const [hover,setHover] = useState<Hit|null>(null);
  const [mode,setMode] = useState<Adjustment["mode"]>("local");
  const [frameIndex,setFrameIndex] = useState(0);
  const [playing,setPlaying] = useState(false);
  const [mediaReady,setMediaReady] = useState(false);
  const [mediaError,setMediaError] = useState("");
  const [showOverlay,setShowOverlay] = useState(true);
  const [showOriginal,setShowOriginal] = useState(false);
  const [zoom,setZoom] = useState(1);
  const [viewportSize,setViewportSize] = useState<Point>([0,0]);
  const [confirmed,setConfirmed] = useState(recovered?.confirmed ?? initial.confirmed);
  const [saving,setSaving] = useState(false);
  const [error,setError] = useState("");
  const [notice,setNotice] = useState(recovered ? "Restored your unsaved adjustments from this tab." : "");
  const operations = history[position]!;
  const dirty = JSON.stringify(operations) !== JSON.stringify(saved.operations) || confirmed !== saved.confirmed;
  const frame = packet.frames[frameIndex]!;
  const aspect = packet.native_size[0]/packet.native_size[1];
  const fittedWidth = Math.min(viewportSize[0],viewportSize[1]*aspect);
  const displayWidth = fittedWidth*zoom;
  const displayScale = displayWidth ? displayWidth/packet.native_size[0] : 1;
  const frameWarning = frame.warnings.length > 0;
  const active = useMemo(()=>showOriginal ? [] : draft ? [...operations,draft] : operations,[operations,draft,showOriginal]);
  const projected = useMemo(()=>projectMarkings(packet,frame,active),[packet,frame,active]);
  const stroke = confirmed && !dirty && !draft && !frameWarning && !showOriginal ? "var(--chalk)" : "var(--amber)";
  useEffect(()=>onDirty(dirty || saving),[dirty,saving,onDirty]);
  useEffect(()=>{
    try {
      if(dirty)sessionStorage.setItem(storageKey,JSON.stringify({version:1,revision:saved.revision,operations,confirmed}));
      else sessionStorage.removeItem(storageKey);
    }catch { /* Browser storage may be disabled; navigation guards still protect edits. */ }
  },[storageKey,dirty,saved.revision,operations,confirmed]);
  useEffect(()=>{
    if(!viewport.current)return;
    const observer=new ResizeObserver(([entry])=>{if(entry)setViewportSize([entry.contentRect.width,entry.contentRect.height]);});
    observer.observe(viewport.current);
    return ()=>observer.disconnect();
  },[]);
  useEffect(()=>{
    if(!dirty)return;
    const warn=(e:BeforeUnloadEvent)=>{e.preventDefault();e.returnValue="";};
    const guard=(e:MouseEvent)=>{
      if(e.button!==0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey)return;
      const anchor=e.target instanceof Element ? e.target.closest("a[href]") : null;
      if(!(anchor instanceof HTMLAnchorElement) || anchor.target==="_blank" || anchor.pathname===window.location.pathname)return;
      e.preventDefault();e.stopPropagation();setNotice("Save or discard your adjustments before leaving this clip.");
    };
    window.addEventListener("beforeunload",warn);
    document.addEventListener("click",guard,true);
    return ()=>{window.removeEventListener("beforeunload",warn);document.removeEventListener("click",guard,true);};
  },[dirty]);
  useEffect(()=>{
    const element=video.current;
    if(!element)return;
    let stopped=false, callback=0;
    const track=(_:number,metadata:VideoFrameCallbackMetadata)=>{
      if(stopped)return;
      setFrameIndex(frameAt(metadata.mediaTime,packet.fps,packet.frames.length));
      callback=element.requestVideoFrameCallback(track);
    };
    if("requestVideoFrameCallback" in element)callback=element.requestVideoFrameCallback(track);
    return ()=>{stopped=true;if(callback)element.cancelVideoFrameCallback(callback);};
  },[packet]);
  const pause=useCallback(()=>{video.current?.pause();setPlaying(false);},[]);
  const seek=useCallback((index:number)=>{
    pause();setHover(null);
    const next=Math.max(0,Math.min(packet.frames.length-1,index));
    if(video.current)video.current.currentTime=(next+.05)/packet.fps;
    // Geometry changes only once seeked/requestVideoFrameCallback reports the frame.
  },[packet,pause]);
  const undo=useCallback(()=>{if(!saving && position>0){pause();setPosition(n=>n-1);setConfirmed(false);setNotice("");setError("");setHover(null);}},[position,saving,pause]);
  const redo=useCallback(()=>{if(!saving && position<history.length-1){pause();setPosition(n=>n+1);setConfirmed(false);setNotice("");setError("");setHover(null);}},[position,history.length,saving,pause]);
  const togglePlay=useCallback(()=>{
    if(!video.current || saving || !mediaReady)return;
    setMediaError("");
    if(!video.current.paused){pause();return;}
    setHover(null);
    if(video.current.ended)video.current.currentTime=0;
    video.current.play().then(()=>setPlaying(true)).catch(()=>setMediaError("Playback could not start. Try pressing Play again."));
  },[saving,mediaReady,pause]);
  useEffect(()=>{
    const key=(e:KeyboardEvent)=>{
      if(e.target instanceof HTMLElement && /INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName))return;
      if(saving || drag.current)return;
      if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="z"){e.preventDefault();if(e.shiftKey)redo();else undo();}
      else if(e.code==="Space"){e.preventDefault();togglePlay();}
      else if(e.key==="ArrowLeft"){e.preventDefault();seek(frameIndex-1);}
      else if(e.key==="ArrowRight"){e.preventDefault();seek(frameIndex+1);}
    };
    window.addEventListener("keydown",key);
    return ()=>window.removeEventListener("keydown",key);
  },[saving,undo,redo,togglePlay,seek,frameIndex]);
  function nativePoint(e:ReactPointerEvent<SVGSVGElement>):Point {
    const rect=e.currentTarget.getBoundingClientRect();
    return [(e.clientX-rect.left)*packet.native_size[0]/rect.width,(e.clientY-rect.top)*packet.native_size[1]/rect.height];
  }
  function nearest(point:Point):Hit|null {
    const scale=(svg.current?.getBoundingClientRect().width ?? packet.native_size[0])/packet.native_size[0];
    return nearestMarking(packet,projected,point,20/scale);
  }

  function pointerDown(e:ReactPointerEvent<SVGSVGElement>) {
    if(e.button!==0 || saving || !showOverlay || showOriginal || !mediaReady || frameWarning || operations.length>=100)return;
    if(playing){pause();return;}
    if(video.current?.seeking)return;
    const pointer=nativePoint(e), hit=nearest(pointer);
    if(!hit)return;
    e.preventDefault();pause();setNotice("");setError("");
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current={hit,pointer,frameIndex,mode,operation:null};setHover(hit);
  }
  function pointerMove(e:ReactPointerEvent<SVGSVGElement>) {
    const pointer=nativePoint(e), current=drag.current;
    if(!current){if(!playing && showOverlay && !showOriginal && !saving)setHover(nearest(pointer));return;}
    const to=dragTarget(current.hit.pixel,current.pointer,pointer,packet.native_size);
    current.operation=makeAdjustment(packet.frames[current.frameIndex]!,current.frameIndex,current.mode,current.hit.world,current.hit.pixel,to);
    setDraft(current.operation);
  }
  function endDrag(e:ReactPointerEvent<SVGSVGElement>,cancel=false) {
    const current=drag.current;drag.current=null;setDraft(null);setHover(null);
    if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId);
    if(cancel || !current?.operation)return;
    const op=current.operation;
    if(Math.hypot(op.anchor.to_native[0]-op.anchor.from_native[0],op.anchor.to_native[1]-op.anchor.from_native[1])<.25)return;
    setHistory([...history.slice(0,position+1),[...operations,op]]);setPosition(position+1);setConfirmed(false);
  }
  async function save() {
    pause();setSaving(true);setError("");setNotice("");setHover(null);
    try {
      const state=await getJSON<AlignmentState>(`/api/alignment/${packet.id}/state`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({expected_revision:saved.revision,source_sha256:packet.source_sha256,base_sha256:packet.base_sha256,operations,confirmed})});
      setSaved(state);setHistory(prior=>prior.map((ops,i)=>i===position?state.operations:ops));setConfirmed(state.confirmed);setNotice("Alignment saved on this computer.");
    }catch(e){setError(e instanceof Error ? e.message : "The alignment could not be saved.");}finally{setSaving(false);}
  }
  const focusPixel=hover ? projected[hover.marking]?.[hover.sample] : null;
  return <div className={styles.editor}>
    <div className={styles.viewerColumn}>
      <div className={styles.viewerToolbar}>
        <span className={styles.reviewState}>{showOriginal ? "Original alignment" : dirty || draft ? "Unsaved adjustments" : confirmed ? "Visually confirmed" : "Ready for your review"}</span>
        <div className={styles.viewOptions}>
          <label><input type="checkbox" checked={showOverlay} disabled={saving} onChange={e=>{setShowOverlay(e.target.checked);setHover(null);}}/>Field lines</label>
          <button aria-pressed={showOriginal} disabled={saving || (!operations.length && !showOriginal)} onClick={()=>{setShowOriginal(v=>!v);setHover(null);}}>{showOriginal?"Show adjusted":"Compare original"}</button>
          <label className={styles.zoom}>Zoom<select aria-label="Video zoom" value={zoom} onChange={e=>setZoom(Number(e.target.value))}><option value={1}>Fit</option><option value={1.5}>150%</option><option value={2}>200%</option><option value={3}>300%</option></select></label>
        </div>
      </div>
      <div ref={viewport} className={styles.viewport}>
        <div className={styles.videoSurface} style={{width:displayWidth || "100%",marginTop:Math.max(0,(viewportSize[1]-displayWidth/aspect)/2)}}>
          <video ref={video} src={`/api/alignment/${packet.id}/video`} preload="auto" playsInline muted aria-label={`${packet.title} field alignment review`} onLoadedData={()=>{setMediaReady(true);setMediaError("");}} onPlay={()=>{setPlaying(true);setMediaError("");}} onPause={()=>setPlaying(false)} onEnded={()=>setPlaying(false)} onSeeked={()=>{if(video.current)setFrameIndex(frameAt(video.current.currentTime,packet.fps,packet.frames.length));}} onTimeUpdate={()=>{const element=video.current;if(element && typeof element.requestVideoFrameCallback!=="function")setFrameIndex(frameAt(element.currentTime,packet.fps,packet.frames.length));}} onError={()=>setMediaError("The review video could not load. Refresh the page to try again.")}/>
          <svg ref={svg} viewBox={`0 0 ${packet.native_size[0]} ${packet.native_size[1]}`} className={`${styles.overlay} ${hover?styles.canDrag:""}`} aria-label="Drag a visible field marking onto the white paint" onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={e=>endDrag(e)} onPointerCancel={e=>endDrag(e,true)} onLostPointerCapture={e=>{if(drag.current)endDrag(e,true);}} onPointerLeave={()=>{if(!drag.current)setHover(null);}}>
            {showOverlay ? projected.map((points,index)=><path key={packet.markings[index]!.id} d={markingPath(points)} fill="none" stroke={stroke} strokeWidth={1.8} vectorEffect="non-scaling-stroke" opacity={.9}/>) : null}
            {showOverlay && focusPixel ? <><circle cx={focusPixel[0]} cy={focusPixel[1]} r={8/displayScale} fill="var(--bg-0)" stroke={stroke} strokeWidth={2} vectorEffect="non-scaling-stroke"/><circle cx={focusPixel[0]} cy={focusPixel[1]} r={2.5/displayScale} fill={stroke}/></> : null}
          </svg>
        </div>
        {!mediaReady && !mediaError ? <div className={styles.videoMessage} role="status">Loading video…</div> : null}
        {mediaError ? <div className={styles.videoMessage} role="alert">{mediaError}</div> : null}
      </div>
      <div className={styles.transport}>
        <button className={styles.play} onClick={togglePlay} disabled={!mediaReady || saving}>{playing?"Pause":"Play clip"}</button>
        <button className={styles.frameButton} onClick={()=>seek(frameIndex-1)} disabled={saving || !mediaReady || frameIndex===0} aria-label="Previous frame" title="Previous frame (left arrow)"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 4v12m10-12-8 6 8 6z"/></svg></button>
        <button className={styles.frameButton} onClick={()=>seek(frameIndex+1)} disabled={saving || !mediaReady || frameIndex===packet.frames.length-1} aria-label="Next frame" title="Next frame (right arrow)"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M15 4v12M5 4l8 6-8 6z"/></svg></button>
        <label className={styles.timeline}><span className={styles.srOnly}>Position in review clip</span><input aria-valuetext={`${clockTime(frame.source_seconds)} in source video`} type="range" min={0} max={packet.frames.length-1} step={1} value={frameIndex} disabled={!mediaReady || saving} onChange={e=>seek(Number(e.target.value))}/></label>
        <span className={`mono ${styles.clock}`}>{clockTime(frame.source_seconds)}<span> / {clockTime(packet.frames.at(-1)!.source_seconds)}</span></span>
      </div>
      <div className={styles.caption}><span>{hover ? markingName(packet.markings[hover.marking]!.id) : playing ? "Checking the alignment through camera motion" : "Pause on a clear view. Drag any visible field line to adjust it."}</span><span>Source video time · {packet.fps} fps review</span></div>
    </div>
    <aside className={styles.dock} aria-label="Alignment controls">
      <div className={styles.tools}>
        <h2>Make it line up</h2>
        <p>Pick a clear line or corner and drag the overlay onto the white paint.</p>
        <fieldset disabled={saving}><legend>Adjustment</legend>
          <label className={mode==="local"?styles.selectedMode:""}><input type="radio" name="adjustment-mode" checked={mode==="local"} onChange={()=>setMode("local")}/><span><strong>Adjust here</strong><small>Move a line or corner nearby.</small></span></label>
          <label className={mode==="global"?styles.selectedMode:""}><input type="radio" name="adjustment-mode" checked={mode==="global"} onChange={()=>setMode("global")}/><span><strong>Move field</strong><small>Shift the whole overlay.</small></span></label>
        </fieldset>
        <div className={styles.undoRow}><button onClick={undo} disabled={saving || position===0}>Undo</button><button onClick={redo} disabled={saving || position===history.length-1}>Redo</button></div>
        {dirty ? <button className={styles.textButton} disabled={saving} onClick={()=>{pause();setHistory([...history.slice(0,position+1),saved.operations]);setPosition(position+1);setConfirmed(saved.confirmed);setError("");setNotice("");}}>Discard unsaved changes</button> : null}
        {operations.length>=100 ? <p role="status">Undo an adjustment before adding another. This clip allows up to 100 adjustments.</p> : null}
      </div>
      <div className={styles.saveArea}>
        {frameWarning ? <p className={styles.warning} role="status">This frame has a mapping warning. Move to another point in the clip before adjusting it.</p> : null}
        <label className={styles.confirm}><input type="checkbox" checked={confirmed} disabled={saving || frameWarning} onChange={e=>setConfirmed(e.target.checked)}/><span>I checked this clip in motion.</span></label>
        <button className={styles.save} disabled={saving || !dirty || !mediaReady} onClick={save}>{saving?"Checking and saving…":"Save alignment"}</button>
        <div aria-live="polite" className={styles.saveStatus}>{notice || (saving ? `Checking the mapping across all ${packet.frames.length} frames.` : dirty ? "Changes stay in this clip. Save before switching clips." : "Saved locally. You can return and keep adjusting.")}</div>
        {error ? <div className={styles.error} role="alert"><p>{error}</p>{error.includes("another tab") ? <button onClick={()=>window.location.reload()}>Reload saved alignment</button> : <p>Your draft is still here. Undo the last adjustment and try saving again.</p>}</div> : null}
      </div>
      <p className={styles.scope}>120 × 70 yd field<br/>Approximate alignment · {packet.duration_s.toFixed(1)} seconds covered.<br/>Unsupported line extensions are hidden.<br/>Visual confirmation does not certify distances.</p>
    </aside>
  </div>;
}
