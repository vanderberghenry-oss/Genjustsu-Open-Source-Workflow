import json,tempfile,unittest
from pathlib import Path
from character_lock import build,load_pack,retarget_points

class IdentityPack(unittest.TestCase):
 def test_reference_or_map_tampering_invalidates_versioned_pack(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ref=root/'face.png';ref.write_bytes(b'fixture-image')
   pack=build(ref,root/'pack','Ma','blue-v1',mapper=lambda *_:{'landmarks':[{'x':.5,'y':.5,'z':0}]})
   self.assertEqual(load_pack(root/'pack')['fingerprint'],pack['fingerprint'])
   with self.assertRaises(ValueError):build(ref,root/'pack','Ma','blue-v1',mapper=lambda *_:{})
   mapped=root/'pack'/'face-map.json';before=mapped.read_bytes();mapped.write_text('{}')
   with self.assertRaises(ValueError):load_pack(root/'pack')
   mapped.write_bytes(before);(root/'pack'/'identity.png').write_bytes(b'changed')
   with self.assertRaises(ValueError):load_pack(root/'pack')
 def test_revision_and_wardrobe_are_part_of_identity_binding(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ref=root/'face.png';ref.write_bytes(b'fixture-image')
   a=build(ref,root/'a','Ma','blue-v1',mapper=lambda *_:{})
   b=build(ref,root/'b','Ma','blue-v2',mapper=lambda *_:{})
   self.assertNotEqual(a['fingerprint'],b['fingerprint'])
   self.assertEqual(a['status'],'mapped-awaiting-identity-review')

class Retarget(unittest.TestCase):
 def points(self):
  p=[{'x':.45+(i%10)*.01,'y':.45+(i//10)*.001} for i in range(478)]
  p[33]={'x':.4,'y':.4};p[263]={'x':.6,'y':.4};return p
 def test_identity_geometry_and_source_translation_have_separate_roles(self):
  p=self.points();target=[dict(v) for v in p];target[1]['y']+=.02
  moved=[{'x':v['x']+.05,'y':v['y']+.03} for v in p]
  result=retarget_points(target,[{'frame':0,'faces':[p]},{'frame':1,'faces':[moved]}])
  self.assertAlmostEqual(result[0]['faces'][0][1]['y'],target[1]['y'])
  self.assertAlmostEqual(result[1]['faces'][0][1]['x']-result[0]['faces'][0][1]['x'],.05)
  self.assertAlmostEqual(result[1]['faces'][0][1]['y']-result[0]['faces'][0][1]['y'],.03)
 def test_bad_geometry_refuses_output(self):
  p=self.points();p[263]=p[33]
  with self.assertRaises(ValueError):retarget_points(p,[{'faces':[p]}])
 def test_reference_expression_does_not_override_protected_source_features(self):
  p=self.points();target=[dict(v) for v in p];target[1]['y']+=.1
  result=retarget_points(target,[{'faces':[p]}],preserve_indices=[1])
  self.assertEqual(result[0]['faces'][0][1],p[1])

if __name__=='__main__':unittest.main()
