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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kshorts-frontend.onrender.com"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
    """Rank multilingual transcript moments using explainable engagement signals."""
    if not segments or duration <= 0:
        return []

    hook_terms = [
        "because", "secret", "mistake", "important", "why", "how", "never",
        "truth", "believe", "remember", "finally", "best", "worst", "surprise",
        "लेकिन", "क्यों", "कैसे", "सच", "गलती", "जरूरी", "याद", "क्योंकि",
        "देखिए", "सोचिए", "ਸੱਚ", "ਕਿਉਂ", "ਕਿਵੇਂ", "ਗਲਤੀ", "ਜ਼ਰੂਰੀ",
        "ਯਾਦ", "ਕਿਉਂਕਿ", "ਵੇਖੋ", "ਸੋਚੋ"
    ]
    emotion_terms = [
        "amazing", "shocking", "love", "hate", "fail", "win", "success",
        "danger", "crazy", "powerful", "incredible", "ध्यान", "कमाल",
        "खतरनाक", "प्यार", "जीत", "ਸਫਲਤਾ", "ਕਮਾਲ", "ਖਤਰਨਾਕ"
    ]

    candidates = []
    for i, first in enumerate(segments):
        start = max(0.0, float(first.get("start", 0)))
        if start >= duration:
            continue
        window = []
        end = start
        for seg in segments[i:]:
            seg_start = float(seg.get("start", 0))
            seg_end = min(duration, float(seg.get("end", seg_start)))
            if seg_start - start > max_len:
                break
            if seg_end > start:
                window.append(seg)
                end = max(end, seg_end)
            if end - start >= 40:
                break
        if end - start < min_len:
            continue
        end = min(duration, end, start + max_len)
        text = clean_text(" ".join(str(x.get("text", "")) for x in window))
        words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        word_count = len(words)
        if word_count < 18:
            continue

        lower_text = text.lower()
        hook_hits = sum(1 for term in hook_terms if term in lower_text)
        emotion_hits = sum(1 for term in emotion_terms if term in lower_text)
        question_hits = text.count("?") + text.count("؟")
        exclamation_hits = text.count("!") + text.count("！")
        number_hits = len(re.findall(r"\d+", text))
        density = word_count / max(1.0, end - start)
        density_score = max(0.0, 3.0 - abs(density - 2.5))
        confidences = [
            max(0.0, min(1.0, float(x.get("confidence", 0.5))))
            for x in window
        ]
        confidence_score = sum(confidences) / max(1, len(confidences))
        completeness = 1.5 if re.search(r"[.!?।؟]$", text) else 0.0
        score = (
            min(word_count, 110) / 18.0
            + min(hook_hits, 5) * 1.15
            + min(emotion_hits, 3) * 0.8
            + min(question_hits, 2) * 0.9
            + min(exclamation_hits, 2) * 0.35
            + min(number_hits, 3) * 0.25
            + density_score + confidence_score + completeness
        )
        candidates.append({
            "start": round(start, 2), "end": round(end, 2),
            "text": text, "score": round(score, 3)
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    chosen = []
    for candidate in candidates:
        if all(
            candidate["end"] + 3 <= item["start"]
            or candidate["start"] >= item["end"] + 3
            for item in chosen
        ):
            chosen.append(candidate)
        if len(chosen) >= max(1, int(wanted)):
            break

    if not chosen:
        start = max(0.0, float(segments[0].get("start", 0)))
        end = min(duration, start + max_len)
        chosen = [{
            "start": start, "end": end,
            "text": clean_text(" ".join(str(x.get("text", "")) for x in segments)),
            "score": 0
        }]
    return sorted(chosen, key=lambda item: item["start"])
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
