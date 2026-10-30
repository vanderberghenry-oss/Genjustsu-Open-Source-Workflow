"""Offline HBC guidance preparation. Never imports provider submission or hosting.

Usage: .venv/bin/python hbc_prepare.py SOURCE EMPTY_JOB_DIRECTORY
Requires the isolated .venv-face worker. Preserves source dialogue without pitch changes.
"""
import json, shutil, subprocess, sys, hashlib, os
from pathlib import Path
from pipeline import normalize
from audio_workflow import read_audio, write_audio, prepare_seedance_video

def prepare(source, folder, track=None):
    source, folder = Path(source).resolve(), Path(folder).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Choose a new empty job directory; existing jobs are preserved.')
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, folder/'upload.mp4')
    normalize(folder/'upload.mp4', folder/'original.mp4')
    if track is None:
        root=Path(__file__).resolve().parent
        python=root/('.venv-face/Scripts/python.exe' if os.name=='nt' else '.venv-face/bin/python')
        if not python.is_file():
            raise ValueError('Install the isolated face worker with setup-face-mesh.sh first.')
        subprocess.run([str(python),str(root/'mesh_worker.py'),str(folder)],check=True,timeout=600,capture_output=True)
    else:
        track(folder)
    tracking=json.loads((folder/'face-tracking.json').read_text())
    if not tracking.get('frames') or tracking.get('missing_frames') or tracking.get('tracked')!=tracking['frames']:
        raise ValueError('Face tracking incomplete. Review and repair before use.')
    audio=read_audio(folder/'upload.mp4')
    # Own performance dialogue is passed through. No Demucs, Rubber Band or remote model.
    write_audio(folder/'vocals.wav',audio)
    (folder/'audio-manifest.json').write_text(json.dumps({'audio_policy':'original-dialogue-no-pitch-shift','separation':'none; use a clean source dialogue recording','pitch_semitones':0}))
    prepared=prepare_seedance_video(folder,semitones=0)
    (folder/'workflow-mode.json').write_text(json.dumps({'mode':'face_mesh'}))
    from mask_review import fingerprint
    result={'mode':'face_mesh','audio_policy':'original-dialogue-no-pitch-shift',
        'guidance_fingerprint':fingerprint(folder),'prepared':str(prepared),
        'prepared_sha256':hashlib.sha256(prepared.read_bytes()).hexdigest(),
        'status':'awaiting-full-clip-review'}
    (folder/'hbc-preparation.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('Usage: hbc_prepare.py SOURCE EMPTY_JOB_DIRECTORY')
    try: print(json.dumps(prepare(sys.argv[1],sys.argv[2])))
    except Exception as exc:
        print(json.dumps({'error':type(exc).__name__,'message':'Local preparation failed; inspect local tracking/dependency diagnostics.'}),file=sys.stderr)
        raise SystemExit(1)
