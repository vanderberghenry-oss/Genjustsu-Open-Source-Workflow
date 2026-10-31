"""Offline HBC guidance preparation. Never imports provider submission or hosting.

Usage:
  .venv/bin/python hbc_prepare.py SOURCE EMPTY_JOB_DIRECTORY [--mode M] [--semitones N] [--subject S]

Modes:
  face_mesh   MediaPipe landmarks only; no depth map, no SAM call (fast, local).
  depth       Coloured depth + SAM 3 raw masks + composite (the repo's advanced path).
  depth_mesh  The depth composite with the face mesh overlaid on it.

Audio: the source dialogue is passed through. `--semitones N` shifts it with
Rubber Band; the standalone CLI is used when present, otherwise FFmpeg's
`rubberband` filter, which links the same library. N=0 keeps the dialogue
untouched, which is the HBC default.
"""
import json, shutil, subprocess, sys, hashlib, os, argparse
from pathlib import Path
try:
    from config import load_env
    load_env()  # reads the worker's .env, where REPLICATE_API_TOKEN belongs
except Exception:
    pass
from pipeline import normalize
from audio_workflow import read_audio, write_audio, prepare_seedance_video

MODES = ('face_mesh', 'depth', 'depth_mesh')


def _face_python(root):
    return root / ('.venv-face/Scripts/python.exe' if os.name == 'nt' else '.venv-face/bin/python')


def prepare(source, folder, mode='face_mesh', semitones=0, subject='person', track=None):
    source, folder = Path(source).resolve(), Path(folder).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Choose a new empty job directory; existing jobs are preserved.')
    folder.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    shutil.copyfile(source, folder / 'upload.mp4')
    normalize(folder / 'upload.mp4', folder / 'original.mp4')

    if mode == 'face_mesh':
        if track is not None:
            track(folder)
        else:
            python = _face_python(root)
            if not python.is_file():
                raise ValueError('Install the isolated face worker with setup-face-mesh.sh first.')
            subprocess.run([str(python), str(root / 'mesh_worker.py'), str(folder)],
                           check=True, timeout=600, capture_output=True)
        tracking = json.loads((folder / 'face-tracking.json').read_text())
        if not tracking.get('frames') or tracking.get('missing_frames') or tracking.get('tracked') != tracking['frames']:
            raise ValueError('Face tracking incomplete. Review and repair before use.')
    else:
        # The repo's own stages: coloured depth, SAM 3 raw masks, composite.
        # Needs REPLICATE_API_TOKEN in the environment; Replicate bills these runs.
        from pipeline import depth_replicate, composite
        from raw_masks import segment_raw
        depth_replicate(folder / 'original.mp4', folder / 'depth.mp4')
        masks = segment_raw(folder / 'original.mp4', folder, subject)
        composite(folder / 'original.mp4', folder / 'depth.mp4', masks, folder, folder / 'upload.mp4')
        if mode == 'depth_mesh':
            python = _face_python(root)
            if not python.is_file():
                raise ValueError('Install the isolated face worker with setup-face-mesh.sh first.')
            shutil.copy2(folder / 'composite-silent.mp4', folder / 'depth-composite-silent.mp4')
            subprocess.run([str(python), str(root / 'mesh_worker.py'), str(folder), '--depth'],
                           check=True, timeout=600, capture_output=True)

    # HBC audio policy: the source dialogue, not a Demucs separation. Pitched only on request.
    audio = read_audio(folder / 'upload.mp4')
    write_audio(folder / 'vocals.wav', audio)
    policy = 'original-dialogue-no-pitch-shift' if semitones == 0 else 'pitched-vocals'
    (folder / 'audio-manifest.json').write_text(json.dumps({
        'audio_policy': policy,
        'separation': 'none; clean source dialogue recording',
        'pitch_semitones': semitones,
    }))
    prepared = prepare_seedance_video(folder, semitones=semitones)
    manifest = json.loads((folder / 'audio-manifest.json').read_text())
    (folder / 'workflow-mode.json').write_text(json.dumps({'mode': mode, 'pitch_semitones': semitones}))
    from mask_review import fingerprint
    result = {
        'mode': mode,
        'audio_policy': policy,
        'pitch_semitones': semitones,
        'pitch_engine': manifest.get('pitch_engine', 'none'),
        'guidance_fingerprint': fingerprint(folder),
        'prepared': str(prepared),
        'prepared_sha256': hashlib.sha256(prepared.read_bytes()).hexdigest(),
        'status': 'awaiting-full-clip-review',
    }
    (folder / 'hbc-preparation.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Offline HBC guidance preparation.')
    parser.add_argument('source')
    parser.add_argument('folder')
    parser.add_argument('--mode', choices=MODES, default='face_mesh')
    parser.add_argument('--semitones', type=float, default=0)
    parser.add_argument('--subject', default='person')
    args = parser.parse_args()
    try:
        semitones = int(args.semitones) if float(args.semitones).is_integer() else round(args.semitones, 2)
        print(json.dumps(prepare(args.source, args.folder, args.mode, semitones, args.subject)))
    except Exception as exc:
        print(json.dumps({'error': type(exc).__name__, 'message': str(exc)[:400] or 'Local preparation failed.'}), file=sys.stderr)
        raise SystemExit(1)
