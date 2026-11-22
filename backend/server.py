import os, uuid, json, shutil, subprocess, threading, re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel

ROOT=Path(os.getenv("DATA_DIR","./data")).resolve()
UPLOADS=ROOT/"uploads"; OUTPUTS=ROOT/"outputs"
for p in (UPLOADS,OUTPUTS): p.mkdir(parents=True,exist_ok=True)
DB=ROOT/"jobs.json"
MAX_BYTES=int(os.getenv("MAX_UPLOAD_MB","800"))*1024*1024
MODEL=os.getenv("WHISPER_MODEL","small")
app=FastAPI(title="K-SHORTS AI API",version="0.1.0")
@app.get("/")
def home():
    return {
        "message": "Welcome to K-SHORTS AI",
        "status": "running",
        "version": "1.0.0"
    }
@app.get("/healthz")
def healthz():
    return {"status": "ok"}
app.add_middleware(CORSMiddleware,allow_origins=os.getenv("CORS_ORIGINS","http://localhost:3000,http://127.0.0.1:8000").split(","),allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
_lock=threading.Lock()
_model=None

def read_db():
    if not DB.exists(): return {}
    try: return json.loads(DB.read_text())
    except Exception: return {}
def write_job(job):
    with _lock:
        db=read_db(); db[job["id"]]=job; DB.write_text(json.dumps(db,indent=2))
def get_job(job_id):
    job=read_db().get(job_id)
    if not job: raise HTTPException(404,"Job not found")
    return job
def run(cmd):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if p.returncode: raise RuntimeError((p.stderr or p.stdout)[-3500:])
def model():
    global _model
    if _model is None:
        _model=WhisperModel(MODEL,device=os.getenv("WHISPER_DEVICE","cpu"),compute_type=os.getenv("WHISPER_COMPUTE","int8"))
    return _model
def srt_time(sec):
    ms=int(max(0,sec)*1000); h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000); s,ms=divmod(ms,1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"
def clean_text(t): return re.sub(r"\s+"," ",t).strip()
def pick_windows(segments, duration, wanted=5, min_len=22, max_len=58):
    # Score transcript windows for complete, expressive phrases; then select non-overlapping moments.
    candidates=[]
    for i,seg in enumerate(segments):
        start=float(seg["start"]); end=min(duration,start+max_len)
        if end-start<min_len: continue
        window=[x for x in segments if x["start"]>=start and x["start"]<end]
        text=" ".join(x["text"] for x in window).strip()
        words=len(text.split())
        if words<18: continue
        cues=sum(1 for w in ["because","secret","mistake","important","why","how","लेकिन","क्यों","कैसे","ਸੱਚ","ਕਿਉਂ","ਕਿਵੇਂ"] if w.lower() in text.lower())
        score=min(words,100)/12 + cues*1.8 + min(1.0,sum(x.get("confidence",0.5) for x in window)/max(1,len(window)))
        candidates.append({"start":start,"end":end,"text":text,"score":score})
    candidates.sort(key=lambda x:x["score"],reverse=True)
    chosen=[]
    for c in candidates:
        if all(c["end"]<=x["start"] or c["start"]>=x["end"] for x in chosen):
            chosen.append(c)
        if len(chosen)>=wanted: break
    if not chosen and duration>0:
        start=0; chosen=[{"start":start,"end":min(duration,max_len),"text":"","score":0}]
    return sorted(chosen,key=lambda x:x["start"])
def process_job(job_id,src,language,count,aspect):
    job=get_job(job_id); job.update(status="processing",progress=5,message="Preparing audio"); write_job(job)
    try:
        probe=subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","json",src],text=True)
        duration=float(json.loads(probe)["format"]["duration"])
        job.update(progress=12,message="Transcribing speech"); write_job(job)
        lang=None if language.lower() in ("auto","auto detect","") else language[:2].lower()
        segments_iter,info=model().transcribe(src,language=lang,word_timestamps=False,vad_filter=True)
        segments=[]
        for s in segments_iter:
            segments.append({"start":float(s.start),"end":float(s.end),"text":clean_text(s.text),"confidence":float(getattr(s,"avg_logprob",-.5)+1)})
        if not segments: raise RuntimeError("No speech detected. Try a video with clearer audio.")
        windows=pick_windows(segments,duration,int(count))
        if not windows: raise RuntimeError("Could not find suitable speech segments in this video.")
        results=[]
        for idx,w in enumerate(windows,1):
            start=max(0,w["start"]); end=min(duration,w["end"]); clip_id=f"{job_id}_{idx}"
            srt=OUTPUTS/f"{clip_id}.srt"; out=OUTPUTS/f"{clip_id}.mp4"
            lines=[]
            relevant=[s for s in segments if s["end"]>start and s["start"]<end]
            for n,s in enumerate(relevant,1):
                a=max(0,s["start"]-start); b=min(end-start,s["end"]-start)
                if b<=a: continue
                lines += [str(n),f"{srt_time(a)} --> {srt_time(b)}",s["text"],""]
            srt.write_text("\n".join(lines),encoding="utf-8")
            vf="scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
            if aspect=="1:1": vf="scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080"
            elif aspect=="16:9": vf="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
            # FFmpeg subtitles filter uses a filesystem path; normalize for filter syntax.
            subpath=str(srt).replace("\\","/").replace(":","\\:")
            vf += f",subtitles='{subpath}':force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00101010,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=110'"
            run(["ffmpeg","-y","-ss",str(start),"-i",src,"-t",str(end-start),"-vf",vf,"-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k","-movflags","+faststart",str(out)])
            results.append({"id":clip_id,"filename":out.name,"download_url":f"/api/clips/{out.name}","start":round(start,2),"end":round(end,2),"duration":round(end-start,2),"transcript":w["text"][:500]})
            job.update(progress=20+int(75*idx/len(windows)),message=f"Rendering clip {idx} of {len(windows)}",clips=results); write_job(job)
        job.update(status="completed",progress=100,message="Your clips are ready",clips=results,completed_at=datetime.now(timezone.utc).isoformat()); write_job(job)
    except Exception as e:
        job.update(status="failed",message=str(e)[:1000],progress=100); write_job(job)
    finally:
        try: os.remove(src)
        except OSError: pass

@app.get("/health")
def health(): return {"ok":True,"service":"K-SHORTS AI API"}
@app.post("/api/jobs/upload")
async def upload_video(background:BackgroundTasks,file:UploadFile=File(...),language:str=Form("auto"),count:int=Form(5),aspect:str=Form("9:16")):
    ext=Path(file.filename or "").suffix.lower()
    if ext not in {".mp4",".mov",".mkv",".webm",".m4v"}: raise HTTPException(400,"Supported formats: MP4, MOV, MKV, WebM, M4V")
    if count not in (3,5,10): raise HTTPException(400,"Clip count must be 3, 5 or 10")
    job_id=uuid.uuid4().hex
    dest=UPLOADS/f"{job_id}{ext}"
    size=0
    with dest.open("wb") as f:
        while chunk:=await file.read(1024*1024):
            size+=len(chunk)
            if size>MAX_BYTES:
                dest.unlink(missing_ok=True); raise HTTPException(413,f"Upload exceeds {MAX_BYTES//1024//1024} MB limit")
            f.write(chunk)
    job={"id":job_id,"status":"queued","progress":0,"message":"Upload received","filename":file.filename,"created_at":datetime.now(timezone.utc).isoformat(),"clips":[]}
    write_job(job); background.add_task(process_job,job_id,str(dest),language,count,aspect)
    return job
@app.get("/api/jobs/{job_id}")
def job_status(job_id:str): return get_job(job_id)
@app.get("/api/clips/{filename}")
def download_clip(filename:str):
    if Path(filename).name!=filename or not filename.endswith(".mp4"): raise HTTPException(400,"Invalid clip filename")
    path=OUTPUTS/filename
    if not path.exists(): raise HTTPException(404,"Clip not ready")
    return FileResponse(path,media_type="video/mp4",filename=filename)
@app.post("/api/jobs/youtube")
def youtube_import_disabled():
    raise HTTPException(501,"Direct YouTube downloading is not enabled. Use a video you own via upload, or connect an authorized video provider/API that permits source ingestion.")
