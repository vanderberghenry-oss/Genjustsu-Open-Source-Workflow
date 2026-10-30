"""Offline character identity mapping and reproducible shot binding.

Uses the isolated face environment. No hosting, credentials or generation calls.
Landmarks describe the reference; they are not hard constraints on a generator.
"""
import argparse, hashlib, json, shutil
from pathlib import Path

def sha(file):
    h=hashlib.sha256()
    with Path(file).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def empty(folder):
    folder=Path(folder).resolve()
    if folder.exists() and any(folder.iterdir()):raise ValueError('Choose a new empty destination; existing records are preserved.')
    folder.mkdir(parents=True,exist_ok=True)
    return folder

def mapped_face(reference,model):
    import mediapipe as mp
    opts=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),num_faces=2,output_face_blendshapes=True,output_facial_transformation_matrixes=True)
    with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as detector:
        result=detector.detect(mp.Image.create_from_file(str(reference)))
    if len(result.face_landmarks)!=1:raise ValueError('An identity reference must contain exactly one detectable face.')
    points=[{'x':p.x,'y':p.y,'z':p.z} for p in result.face_landmarks[0]]
    return {'landmarks':points,'blendshapes':{c.category_name:c.score for c in result.face_blendshapes[0]},'facial_transform':result.facial_transformation_matrixes[0].tolist()}

def build(reference,folder,name,wardrobe,revision=1,mapper=None):
    reference=Path(reference).resolve();model=Path(__file__).parent/'models/face_landmarker.task'
    if reference.suffix.lower() not in ('.png','.jpg','.jpeg','.webp'):raise ValueError('Choose an image reference.')
    if not isinstance(revision,int) or revision<1:raise ValueError('Identity revision must be positive.')
    if not name.strip() or not wardrobe.strip():raise ValueError('Name and wardrobe version are required.')
    source_hash=sha(reference)
    mapping=(mapper or mapped_face)(reference,model)
    if sha(reference)!=source_hash:raise ValueError('Reference changed during mapping.')
    folder=empty(folder);owned=folder/('identity'+reference.suffix.lower());shutil.copyfile(reference,owned)
    mapping_path=folder/'face-map.json';mapping_path.write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    core={'schema_version':1,'name':name.strip(),'revision':revision,'wardrobe_version':wardrobe.strip(),'reference':{'file':owned.name,'sha256':sha(owned)},'face_map':{'file':mapping_path.name,'sha256':sha(mapping_path)},'mapping_model_sha256':sha(model) if mapper is None else 'fixture','mapping_method':'mediapipe-reference-face-map-v1','status':'mapped-awaiting-identity-review','limitations':['Single image records one observed pose and expression.','No face recognition or identity guarantee.','No 3D identity reconstruction or LoRA training.','Reference expression must not replace performance expression.']}
    pack={**core,'fingerprint':fingerprint(core)}
    (folder/'identity-pack.json').write_text(json.dumps(pack,indent=2),encoding='utf-8')
    return pack

def load_pack(folder):
    folder=Path(folder).resolve();pack=json.loads((folder/'identity-pack.json').read_text(encoding='utf-8'))
    core={k:v for k,v in pack.items() if k!='fingerprint'}
    if fingerprint(core)!=pack['fingerprint']:raise ValueError('Identity pack changed.')
    for key in ('reference','face_map'):
        entry=pack[key];p=folder/entry['file']
        if Path(entry['file']).name!=entry['file'] or p.is_symlink() or sha(p)!=entry['sha256']:raise ValueError('Identity asset changed.')
    return pack

def bind(pack_folder,source_folder,destination):
    pack_folder=Path(pack_folder).resolve();source_folder=Path(source_folder).resolve();pack=load_pack(pack_folder)
    preparation=json.loads((source_folder/'hbc-preparation.json').read_text())
    from mask_review import fingerprint as guidance_fingerprint
    if guidance_fingerprint(source_folder)!=preparation['guidance_fingerprint']:raise ValueError('Performance guidance changed; prepare and review again.')
    tracking=json.loads((source_folder/'face-tracking.json').read_text())
    landmark_record=json.loads((source_folder/'face-landmarks.json').read_text())
    mapped_identity=landmark_record.get('identity_fingerprint')
    if mapped_identity and mapped_identity!=pack['fingerprint']:raise ValueError('Guidance was retargeted for a different identity revision.')
    if not tracking.get('frames') or tracking.get('tracked')!=tracking['frames'] or tracking.get('missing_frames'):raise ValueError('Incomplete performance tracking.')
    if sha(source_folder/'seedance-input.mp4')!=preparation['prepared_sha256']:raise ValueError('Prepared video changed.')
    prompt=f"Replace the entire performer in @Video1 with {pack['name']} from @Image1. Preserve the reference identity, hair and wardrobe ({pack['wardrobe_version']}). Use @Video1 for pose, gaze, expression, hands, action and timing. The reference expression is not a performance instruction. Preserve the source setting, light, composition and camera movement. Remove all cyan mesh lines; they are visual guidance only. No titles, added characters or cuts. Preserve source audio timing."
    core={'schema_version':1,'feature':'character-lock','identity_name':pack['name'],'identity_revision':pack['revision'],'wardrobe_version':pack['wardrobe_version'],'identity_fingerprint':pack['fingerprint'],'identity_map_sha256':pack['face_map']['sha256'],'reference_sha256':pack['reference']['sha256'],'source_sha256':sha(source_folder/'upload.mp4'),'prepared_sha256':preparation['prepared_sha256'],'guidance_fingerprint':preparation['guidance_fingerprint'],'frames':tracking['frames'],'prompt':prompt,'provider_settings':{'model':'seedance_2_5','mode':'video_edit','resolution':'480p','draft':False,'count':1},'review':{'identity':False,'full_clip':False,'paid_submission':False},'status':'prepared-awaiting-review-and-quote','mapping_use':'Reference mesh is stored for inspection and version binding. Provider receives the identity image and performance mesh video; reference mesh is not a hard constraint.'}
    folder=empty(destination)
    if mapped_identity:
        core['mapping_use']='Experimental target-reference 2D mesh shape plus source eye-anchored motion is composited into the video. This is visual guidance, not hard identity conditioning.'
    # Freeze self-contained assets; the next shot always starts from the master image.
    shutil.copyfile(pack_folder/pack['reference']['file'],folder/pack['reference']['file'])
    shutil.copyfile(pack_folder/'face-map.json',folder/'identity-face-map.json')
    shutil.copyfile(source_folder/'seedance-input.mp4',folder/'performance-guidance.mp4')
    record={**core,'fingerprint':fingerprint(core)}
    (folder/'character-lock-shot.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record

def retarget_points(reference,frames,preserve_indices=(),reference_aspect=1,source_aspect=1):
    """Experimental 2D identity shape + source motion, using eye anchors.

    Median source shape is a shot baseline, not a neutral-expression estimate.
    Perspective, large yaw, occlusion and independent depth are not solved here.
    """
    import numpy as np
    def normalized(points,aspect):
        p=np.asarray([[v['x'],v['y']] for v in points],dtype=float)
        if p.shape!=(478,2) or not np.isfinite(p).all():raise ValueError('Expected 478 finite landmarks.')
        p[:,0]*=aspect
        origin=(p[33]+p[263])/2;axis=p[263]-p[33];scale=np.linalg.norm(axis)
        if scale<.01:raise ValueError('Eye anchors are degenerate.')
        axis=axis/scale;basis=np.array([axis,[-axis[1],axis[0]]])
        return (p-origin)@basis.T/scale,(origin,basis,scale)
    target,_=normalized(reference,reference_aspect)
    source=[normalized(f['faces'][0],source_aspect) for f in frames]
    baseline=np.median([p for p,_ in source],axis=0)
    output=[]
    for frame,(motion,(origin,basis,scale)) in zip(frames,source):
        points=(target+motion-baseline)@basis*scale+origin
        points[:,0]/=source_aspect
        if preserve_indices:
            original=np.asarray([[p['x'],p['y']] for p in frame['faces'][0]])
            # Taper displacement around expression anchors rather than joining
            # an unchanged lip/eyelid directly to a displaced mesh neighbour.
            delta=original[:,None,:]-original[list(preserve_indices)][None,:,:]
            delta[:,:,0]*=source_aspect
            distance=np.sqrt((delta**2).sum(axis=2)).min(axis=1)
            weight=np.clip(distance/(scale*.35),0,1)
            points=original+(points-original)*weight[:,None]
        # Keep the performer's mouth, eyelids and brows; a smiling master must
        # not impose its expression on every performance.
        for index in preserve_indices:
            points[index]=[frame['faces'][0][index]['x'],frame['faces'][0][index]['y']]
        if not np.isfinite(points).all() or (points<0).any() or (points>1).any():raise ValueError('Retargeted mesh leaves the frame; manual correction required.')
        output.append({**frame,'faces':[[{'x':float(x),'y':float(y)} for x,y in points]]})
    return output

def render(pack_folder,source_folder,destination):
    import av,mediapipe as mp
    from PIL import ImageDraw
    from fractions import Fraction
    pack_folder=Path(pack_folder);source_folder=Path(source_folder);pack=load_pack(pack_folder)
    source=json.loads((source_folder/'face-landmarks.json').read_text())
    frames=source['frames']
    if source['fps']!=24 or not frames or any(len(f['faces'])!=1 for f in frames):raise ValueError('One tracked face per 24fps frame required.')
    mapping=json.loads((pack_folder/'face-map.json').read_text())
    connections=mp.tasks.vision.FaceLandmarksConnections
    preserve=set()
    for name in ('FACE_LANDMARKS_LIPS','FACE_LANDMARKS_LEFT_EYE','FACE_LANDMARKS_RIGHT_EYE','FACE_LANDMARKS_LEFT_EYEBROW','FACE_LANDMARKS_RIGHT_EYEBROW'):
        for c in getattr(connections,name):preserve.update((c.start,c.end))
    from PIL import Image
    with Image.open(pack_folder/pack['reference']['file']) as ref:reference_aspect=ref.width/ref.height
    with av.open(str(source_folder/'original.mp4')) as video:source_aspect=video.streams.video[0].width/video.streams.video[0].height
    mapped=retarget_points(mapping['landmarks'],frames,preserve,reference_aspect,source_aspect)
    folder=empty(destination)
    for name in ('upload.mp4','original.mp4'):shutil.copyfile(source_folder/name,folder/name)
    with av.open(str(folder/'original.mp4')) as inp,av.open(str(folder/'composite-silent.mp4'),'w') as out:
        stream=out.add_stream('libx264',rate=24);stream.width=inp.streams.video[0].width;stream.height=inp.streams.video[0].height;stream.pix_fmt='yuv420p'
        count=0
        for i,frame in enumerate(inp.decode(video=0)):
            if i>=len(mapped):raise ValueError('Video and tracking length differ.')
            im=frame.to_image();draw=ImageDraw.Draw(im)
            points=[(round(p['x']*im.width),round(p['y']*im.height)) for p in mapped[i]['faces'][0]]
            for c in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:draw.line([points[c.start],points[c.end]],fill=(50,170,205),width=1)
            vf=av.VideoFrame.from_image(im);vf.pts=i;vf.time_base=Fraction(1,24)
            for pkt in stream.encode(vf):out.mux(pkt)
            count+=1
        if count!=len(mapped):raise ValueError('Video and tracking length differ.')
        for pkt in stream.encode():out.mux(pkt)
    (folder/'face-landmarks.json').write_text(json.dumps({'fps':24,'frames':mapped,'identity_fingerprint':pack['fingerprint'],'algorithm':'experimental-2d-eye-frame-median-delta-v2','preserved_source_expression_indices':sorted(preserve),'source_landmarks_sha256':sha(source_folder/'face-landmarks.json')}))
    (folder/'face-tracking.json').write_text(json.dumps({'frames':count,'tracked':count,'missing_frames':[],'note':'Transferred from source tracking; not new detections.'}))
    (folder/'workflow-mode.json').write_text('{"mode":"face_mesh"}')
    (folder/'identity-retarget.json').write_text(json.dumps({'identity_fingerprint':pack['fingerprint'],'reference_sha256':pack['reference']['sha256'],'experimental':True,'approved':False}))
    return {'frames':count,'status':'rendered-awaiting-audio-and-review'}

def finalize(folder):
    from audio_workflow import read_audio,write_audio,prepare_seedance_video
    from mask_review import fingerprint as guidance_fingerprint
    folder=Path(folder)
    if (folder/'hbc-preparation.json').exists():raise ValueError('Preparation already finalized; preserve it.')
    write_audio(folder/'vocals.wav',read_audio(folder/'upload.mp4'))
    (folder/'audio-manifest.json').write_text(json.dumps({'audio_policy':'original-dialogue-no-pitch-shift','pitch_semitones':0,'separation':'none'}))
    prepared=prepare_seedance_video(folder,semitones=0)
    record={'mode':'face_mesh','audio_policy':'original-dialogue-no-pitch-shift','prepared':str(prepared.resolve()),'prepared_sha256':sha(prepared),'guidance_fingerprint':guidance_fingerprint(folder),'status':'awaiting-full-clip-review','experimental_identity_retarget':True}
    (folder/'hbc-preparation.json').write_text(json.dumps(record,indent=2));return record

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    b=sub.add_parser('build');b.add_argument('reference');b.add_argument('folder');b.add_argument('--name',required=True);b.add_argument('--wardrobe',required=True);b.add_argument('--revision',type=int,default=1)
    s=sub.add_parser('bind');s.add_argument('pack');s.add_argument('source_job');s.add_argument('folder')
    s=sub.add_parser('render');s.add_argument('pack');s.add_argument('source_job');s.add_argument('folder')
    s=sub.add_parser('finalize');s.add_argument('folder')
    a=p.parse_args()
    if a.action=='build':result=build(a.reference,a.folder,a.name,a.wardrobe,a.revision)
    elif a.action=='bind':result=bind(a.pack,a.source_job,a.folder)
    elif a.action=='render':result=render(a.pack,a.source_job,a.folder)
    else:result=finalize(a.folder)
    print(json.dumps(result))
