import unittest,tempfile,json,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from audio_workflow import write_audio,read_audio,prepare_seedance_video
from pipeline import writer,put,finish
from hbc_prepare import prepare

class CleanPreparation(unittest.TestCase):
 def test_clean_audio_does_not_call_rubberband_or_network(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)
   # Actual encoded media, not a mocked muxer.
   import av
   from fractions import Fraction
   with av.open(str(p/'composite-silent.mp4'),'w') as out:
    s=out.add_stream('libx264',rate=24);s.width=64;s.height=64;s.pix_fmt='yuv420p'
    for i in range(24):
     f=av.VideoFrame.from_ndarray(np.zeros((64,64,3),dtype=np.uint8),format='rgb24');f.pts=i;f.time_base=Fraction(1,24)
     for packet in s.encode(f):out.mux(packet)
    for packet in s.encode():out.mux(packet)
   t=np.arange(44100)/44100;tone=np.tile((.2*np.sin(2*np.pi*440*t)).astype(np.float32),(2,1));write_audio(p/'vocals.wav',tone)
   with patch('subprocess.run',side_effect=AssertionError('No pitch process')),patch('requests.post',side_effect=AssertionError('No network')):
    result=prepare_seedance_video(p,semitones=0)
   audio=read_audio(result)[0];freq=np.argmax(abs(np.fft.rfft(audio)))*44100/len(audio)
   self.assertLess(abs(freq-440),3)
   self.assertEqual(json.loads((p/'audio-manifest.json').read_text())['pitch_semitones'],0)
   # Full offline preparation uses a fixture tracker; no real face tracking claim.
   dest=p/'job'
   def track(folder):
    shutil.copyfile(folder/'original.mp4',folder/'composite-silent.mp4')
    (folder/'face-tracking.json').write_text(json.dumps({'frames':24,'tracked':24,'missing_frames':[]}))
    (folder/'face-landmarks.json').write_text('{}')
   with patch('requests.post',side_effect=AssertionError('No network')):
    job=prepare(result,dest,track=track)
   self.assertEqual(job['status'],'awaiting-full-clip-review')
   self.assertEqual(job['audio_policy'],'original-dialogue-no-pitch-shift')
   with self.assertRaises(ValueError):prepare(result,dest,track=track)


class CleanPayload(unittest.TestCase):
 def test_clean_face_reference_payload_never_claims_pitched_audio(self):
  import seedance_bridge
  from unittest.mock import Mock
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'workflow-mode.json').write_text('{"mode":"face_mesh"}');(p/'audio-manifest.json').write_text('{"audio_policy":"original-dialogue-no-pitch-shift"}')
   response=Mock();response.json.return_value={'requestId':'clean-fixture'}
   with patch('mask_review.require_approved'),patch('seedance_bridge.credential',return_value='dummy'),patch('seedance_bridge.setting',return_value='https://example.org'),patch('seedance_bridge.upload',return_value='https://example.org/media'),patch('seedance_bridge.requests.post',return_value=response) as post,patch('audio_workflow.prepare_seedance_video',side_effect=AssertionError('Must not mutate reviewed input')):
    seedance_bridge.submit(p,'Replace subject','identity.png')
   payload=post.call_args.kwargs['json'];self.assertIn('no pitch modification',payload['prompt']);self.assertNotIn('three semitones',payload['prompt']);self.assertEqual(payload['audios'],[])

if __name__=='__main__':unittest.main()
