from pathlib import Path
import argparse,hashlib,json,unittest
import numpy as np
from PIL import Image
import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform
MEAN=[.485,.456,.406]
STD=[.229,.224,.225]
LEGACY='legacy_external_plus_internal_imagenet_v1'
STANDARD='torchvision_internal_imagenet_v1'
PROFILES={LEGACY,STANDARD}
def resize_rgb(image,size=640):
 image=image.convert('RGB');scale=size/min(image.size)
 if scale*max(image.size)>1.6*size:scale=1.6*size/max(image.size)
 return image.resize((int(round(image.width*scale)),int(round(image.height*scale))))
def to_model_tensor(resized_rgb,profile):
 if profile not in PROFILES:raise ValueError('Explicit supported input profile required')
 arr=np.asarray(resized_rgb.convert('RGB'),dtype=np.float32)/255.
 if profile==LEGACY:arr=(arr-np.array(MEAN,dtype=np.float32))/np.array(STD,dtype=np.float32)
 return torch.from_numpy(arr.transpose(2,0,1))
def prepare(image,profile,size=640):
 resized=resize_rgb(image,size);return to_model_tensor(resized,profile),resized
def assert_internal_transform(model):
 transform=model.transform
 if transform.image_mean!=MEAN or transform.image_std!=STD:raise ValueError('Unexpected internal normalization; profile is not transferable')
 if tuple(transform.min_size)!=(800,) or transform.max_size!=1333:raise ValueError('Unexpected detector resize contract')
def validate_checkpoint(path,manifest):
 if manifest.get('input_profile') not in PROFILES:raise ValueError('Checkpoint lacks an explicit input profile')
 actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
 if actual!=manifest.get('checkpoint_sha256'):raise ValueError('Checkpoint/profile binding mismatch')
 return manifest['input_profile']
class ContractTests(unittest.TestCase):
 def test_standard_float32_and_range(self):
  x,_=prepare(Image.new('RGB',(16,9),(0,128,255)),STANDARD,16);self.assertEqual(x.dtype,torch.float32);self.assertEqual(float(x.min()),0.);self.assertEqual(float(x.max()),1.)
 def test_legacy_matches_historical_numpy(self):
  im=Image.new('RGB',(16,9),(30,100,200));x,resized=prepare(im,LEGACY,16);a=np.asarray(resized,dtype=np.float32)/255.;a=(a-np.array(MEAN,np.float32))/np.array(STD,np.float32);self.assertTrue(torch.equal(x,torch.from_numpy(a.transpose(2,0,1))))
 def test_aspect_and_cap(self):self.assertEqual(resize_rgb(Image.new('RGB',(1280,720))).size,(1024,576))
 def test_portrait_aspect(self):self.assertEqual(resize_rgb(Image.new('RGB',(720,1280))).size,(576,1024))
 def test_rgb_channels(self):self.assertEqual(prepare(Image.new('L',(20,20)),STANDARD,20)[0].shape[0],3)
 def test_unknown_profile_rejected(self):
  with self.assertRaises(ValueError):prepare(Image.new('RGB',(10,10)),'guess')
 def test_explicit_profile_required(self):
  with self.assertRaises(TypeError):prepare(Image.new('RGB',(10,10)))
 def test_internal_transform_contract(self):
  class M:pass
  m=M();m.transform=GeneralizedRCNNTransform(800,1333,MEAN,STD);assert_internal_transform(m)
 def test_identity_internal_normalization_rejected(self):
  class M:pass
  m=M();m.transform=GeneralizedRCNNTransform(800,1333,[0,0,0],[1,1,1])
  with self.assertRaises(ValueError):assert_internal_transform(m)
 def test_actual_single_double_difference(self):
  t=GeneralizedRCNNTransform(800,1333,MEAN,STD);x=torch.full((3,2,2),.5);once=t.normalize(x);twice=t.normalize(once);self.assertGreater(float((once-twice).abs().max()),1.)
 def test_resize_does_not_crop(self):
  im=Image.new('RGB',(1280,720),'red');x,resized=prepare(im,STANDARD);self.assertEqual(resized.getpixel((0,0)),(255,0,0));self.assertEqual(resized.getpixel((1023,575)),(255,0,0))
 def test_standard_is_not_legacy(self):
  im=Image.new('RGB',(20,20),(128,128,128));a,_=prepare(im,STANDARD,20);b,_=prepare(im,LEGACY,20);self.assertFalse(torch.equal(a,b))
 def test_missing_checkpoint_profile_rejected_before_io(self):
  with self.assertRaises(ValueError):validate_checkpoint('unused',{})
 def test_checkpoint_hash_mismatch_rejected(self):
  with self.assertRaises(ValueError):validate_checkpoint(__file__,{'input_profile':LEGACY,'checkpoint_sha256':'0'*64})
 def test_hash_binding_accepts_exact_fixture_file(self):
  digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();self.assertEqual(validate_checkpoint(__file__,{'input_profile':LEGACY,'checkpoint_sha256':digest}),LEGACY)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
 if a.self_test:unittest.main(argv=['detection_input_contract.py'])
 else:ap.error('Import the explicit input contract or use --self-test; this utility cannot authorize printer release')
