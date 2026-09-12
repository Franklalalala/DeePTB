import unittest,tempfile
from pathlib import Path
import numpy as np
from h0rebuild.assemble import AssemblyResult,hermiticity_report
from h0rebuild.models import BlockKey
from h0rebuild.cell_gauge import rebase_result,shifts_between_coordinates,validate_cell_shifts
from h0rebuild.io import save_result,load_blocks,validate_artifact

class CellGaugeTests(unittest.TestCase):
 def make_result(self):
  rng=np.random.default_rng(61)
  a=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
  blocks={BlockKey(0,1,(1,0,-1)):a,BlockKey(1,0,(-1,0,1)):a.conj().T,
          BlockKey(0,0,(0,0,0)):np.diag([2.,3.]),BlockKey(1,1,(0,0,0)):np.diag([4.,5.])}
  meta={'structure':{'cell_bohr':[[3.,0.,0.],[-1.,4.,0.],[.3,.1,5.]],'species':['X','Y'],'fractional_coordinates':[[-.1,.2,.3],[.4,1.2,.6]]}}
  return AssemblyResult(blocks,dict(blocks),{'kinetic':dict(blocks),'local':dict(blocks),'nonlocal':dict(blocks)},None,[1,1],meta)
 def test_spinor_bloch_and_displacements(self):
  a=self.make_result();q=np.array([[1,0,0],[0,-1,0]]);b=rebase_result(a,q)
  cell=np.array(a.metadata['structure']['cell_bohr']);frac=np.array(a.metadata['structure']['fractional_coordinates'])
  for key,value in a.h_blocks_ry.items():
   rr=np.array(key.R)+q[key.i]-q[key.j];other=BlockKey(key.i,key.j,tuple(rr))
   np.testing.assert_array_equal(b.h_blocks_ry[other],value)
   np.testing.assert_allclose((frac[key.j]+key.R-frac[key.i])@cell,(frac[key.j]+q[key.j]+rr-frac[key.i]-q[key.i])@cell,atol=1e-14)
   for comp in b.components_ry:np.testing.assert_array_equal(b.components_ry[comp][other],value)
  k=np.array([.13,.27,.39])
  def bloch(blocks):
   h=np.zeros((4,4),complex)
   for key,v in blocks.items():h[2*key.i:2*key.i+2,2*key.j:2*key.j+2]+=v*np.exp(2j*np.pi*np.dot(k,key.R))
   return h
  ha,hb=bloch(a.h_blocks_ry),bloch(b.h_blocks_ry);u=np.diag(np.repeat(np.exp(2j*np.pi*(q@k)),2))
  np.testing.assert_allclose(hb,u@ha@u.conj().T,atol=1e-13)
  np.testing.assert_allclose(np.linalg.eigvalsh(ha),np.linalg.eigvalsh(hb),atol=1e-13)
  self.assertEqual(hermiticity_report(b.h_blocks_ry)['max_abs'],0.)
  restored=rebase_result(b,-q)
  for key,v in a.h_blocks_ry.items():np.testing.assert_array_equal(restored.h_blocks_ry[key],v)
  self.assertNotIn('cell_gauge',a.metadata)
 def test_geometry_only_shift_and_reject_displacement(self):
  a=self.make_result();cell=np.array(a.metadata['structure']['cell_bohr']);frac=np.array(a.metadata['structure']['fractional_coordinates']);q=np.array([[1,0,0],[0,-1,0]])
  np.testing.assert_array_equal(shifts_between_coordinates(cell,frac,(frac+q)@cell),q)
  with self.assertRaises(ValueError):shifts_between_coordinates(cell,frac,(frac+q)@cell+.01)
  with self.assertRaises(ValueError):validate_cell_shifts([[.1,0,0],[0,0,0]],2)
  with self.assertRaises(ValueError):validate_cell_shifts([[1,0,0]],2)
 def test_serialization_keeps_rebased_keys_and_geometry(self):
  result=rebase_result(self.make_result(),[[1,0,0],[0,-1,0]])
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'result.npz';save_result(result,path,energy_unit='Ry');meta=validate_artifact(path);loaded=load_blocks(path)
   self.assertEqual(meta['structure'],result.metadata['structure'])
   self.assertEqual(set(loaded),set(result.h_blocks_ry))
   for k,v in loaded.items():np.testing.assert_array_equal(v,result.h_blocks_ry[k])

if __name__=='__main__':unittest.main()
