"""Boundary and interrupted-training tests for the isolated D implementation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.train_next.core import atomic_json, fit_scaler, sha256
from tools.train_next.runtime import run_lease
from tools.train_next_v2.common import BASE, ROOT, process_identity
from tools.train_next_v2.model import SequenceModel
from tools.train_next_v2.tests.test_experiments import songs_fixture, SETTINGS, seed_all
from .data import relative, validate_pool, read_auxiliary_feature, PROTOCOL, SOURCE
from tools.train_next_v2.common import read_json
from tools.train_next_v2.jamendo_data.features import CACHE, read_cached
from tools.train_next_v2.jamendo_data.contract import TAG_ORDER, observed_tag_distribution
from .engine import train_loop, tensor_digest
from .model import AuxiliaryModel, tag_loss
from .sampling import PairedSampler, collate_auxiliary
from . import status as status_module

SETUP = dict(SETTINGS, derivative_weight=0., tag_loss_weight=.1, aux_batch_size=3)


def clips_fixture():
    rng = np.random.default_rng(304)
    return [dict(song_id=f'aux-{i}', group_id=f'artist-{i//2}', observed_tags=[TAG_ORDER[i % 12], TAG_ORDER[(i+1) % 12]],
                 x=rng.normal(size=(3+i, 3)).astype(np.float32)) for i in range(7)]


class AuxiliaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    def setUp(self):
        (BASE/'tmp').mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='auxiliary-test-', dir=BASE/'tmp')
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to((BASE/'tmp').resolve()))

    def tearDown(self):
        self.assertTrue(self.root.is_relative_to((BASE/'tmp').resolve()))
        self.temp.cleanup()

    def folders(self, label):
        pair = (self.root/label/'run', self.root/label/'models')
        for path in pair:
            path.mkdir(parents=True)
        return pair

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_va_matches_reference_and_class_pool_ignores_padding(self):
        seed_all(); reference = SequenceModel(3, SETUP).eval()
        seed_all(); model = AuxiliaryModel(3, SETUP).eval()
        values, lengths = torch.randn(2, 9, 3), torch.tensor([9, 4])
        with torch.inference_mode():
            torch.testing.assert_close(reference(values, lengths), model(values, lengths), rtol=0, atol=0)
            original = model.classify(values, lengths)
            values[1, 4:] = 1e7
            changed = model.classify(values, lengths)
            alone = model.classify(values[1:2, :4], torch.tensor([4]))
        torch.testing.assert_close(original, changed, rtol=0, atol=0)
        torch.testing.assert_close(changed[1], alone[0], rtol=1e-5, atol=1e-6)
        self.assertEqual(original.shape, (2, 12))

    def test_auxiliary_gradient_reaches_backbone_but_not_va_head(self):
        model = AuxiliaryModel(3, SETUP)
        logits = model.classify(torch.randn(2, 7, 3), torch.tensor([7, 3]))
        targets=torch.from_numpy(np.stack([observed_tag_distribution([TAG_ORDER[0]]), observed_tag_distribution([TAG_ORDER[2],TAG_ORDER[3]])]))
        tag_loss(logits,targets).backward()
        self.assertGreater(float(model.lstm.weight_ih_l0.grad.abs().sum()), 0)
        self.assertIsNone(model.head[1].weight.grad)

    def test_aux_collate_has_one_distribution_and_rejects_fake_temporal_targets(self):
        clips = clips_fixture(); mean = np.ones(3, np.float32); std = np.full(3, 2, np.float32)
        values, classes, lengths = collate_auxiliary(clips, mean, std)
        self.assertEqual(classes.shape, (7, 12))
        np.testing.assert_array_equal(values[0, :3].numpy(), (clips[0]['x']-mean)/std)
        self.assertTrue(torch.equal(values[0, 3:], torch.zeros_like(values[0, 3:])))
        with self.assertRaisesRegex(ValueError, 'whole-track'):
            collate_auxiliary([dict(clips[0], y=np.zeros((3, 2)))], mean, std)

    def test_sampler_streams_membership_and_global_rng_independence(self):
        songs, clips = songs_fixture(), clips_fixture()
        sampler = PairedSampler(songs['fit'], clips, SETUP)
        expected = [sampler.indices(s) for s in range(1, 8)]
        random.random(); np.random.random(120); torch.rand(19)
        restored = PairedSampler(songs['fit'], clips, SETUP)
        self.assertEqual(expected[-1], restored.indices(7))
        restored.verify(sampler.state(6), 6)
        altered = dict(sampler.state(6), next_step=6)
        with self.assertRaisesRegex(ValueError, 'sampler'):
            restored.verify(altered, 6)
        different_aux = PairedSampler(songs['fit'], clips[:3], SETUP)
        self.assertEqual(sampler.indices(9)[0], different_aux.indices(9)[0])
        with self.assertRaisesRegex(ValueError, 'membership'):
            different_aux.verify(sampler.state(6), 6)

    def exercise_resume(self, device):
        songs, clips, provenance = songs_fixture(), clips_fixture(), dict(data='synthetic-v1', code='immutable-v1')
        whole, interrupted = self.folders(device.type+'-full'), self.folders(device.type+'-resumed')
        seed_all()
        train_loop(songs, clips, SETUP, provenance, *whole, device)
        def interrupt(step):
            if step == 2:
                raise InterruptedError('Synthetic process exit after full checkpoint')
        seed_all()
        with self.assertRaises(InterruptedError):
            train_loop(songs, clips, SETUP, provenance, *interrupted, device, after_checkpoint=interrupt)
        saved_bytes = (interrupted[1]/'last.pth').read_bytes()
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            train_loop(songs, clips, SETUP, dict(provenance, code='mutated'), *interrupted, device, resume=True)
        self.assertEqual(saved_bytes, (interrupted[1]/'last.pth').read_bytes())
        seed_all()
        train_loop(songs, clips, SETUP, provenance, *interrupted, device, resume=True)
        a, b = [torch.load(p[1]/'last.pth', weights_only=True, map_location='cpu') for p in (whole, interrupted)]
        for key in ('state_dict', 'optimizer', 'mean', 'std', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng',
                    'samplers', 'exposure_chain', 'initial_state_sha256', 'best_state_dict', 'best_score'):
            self.assert_nested_equal(a[key], b[key])
        self.assertEqual(a['samplers']['main_exposures'], 18)
        self.assertEqual(a['samplers']['auxiliary_forward_exposures'], 18)

    def test_full_resume_cpu_is_bitwise_identical(self):
        self.exercise_resume(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CPU preflight; GPU resume is checked before admission')
    def test_full_resume_cuda_is_bitwise_identical(self):
        self.exercise_resume(torch.device('cuda'))

    def test_control_auxiliary_content_cannot_change_va_and_scaler(self):
        songs, clips = songs_fixture(), clips_fixture()
        altered = [dict(row, x=row['x']*100, observed_tags=[TAG_ORDER[(TAG_ORDER.index(t)+2)%12] for t in row['observed_tags']]) for row in clips]
        settings = dict(SETUP, tag_loss_weight=0., max_steps=2)
        outputs = []
        for i, pool in enumerate((clips, altered)):
            dirs = self.folders(str(i)); seed_all()
            train_loop(songs, pool, settings, {}, *dirs, torch.device('cpu'))
            outputs.append(torch.load(dirs[1]/'last.pth', weights_only=True, map_location='cpu'))
        self.assert_nested_equal(outputs[0]['state_dict'], outputs[1]['state_dict'])
        self.assert_nested_equal(outputs[0]['torch_rng'], outputs[1]['torch_rng'])
        mean, std = fit_scaler(songs['fit'])
        np.testing.assert_array_equal(outputs[0]['mean'].numpy(), mean)
        np.testing.assert_array_equal(outputs[0]['std'].numpy(), std)
        auxiliary_dirs = self.folders('aux'); seed_all()
        train_loop(songs, clips, dict(settings, tag_loss_weight=.1), {}, *auxiliary_dirs, torch.device('cpu'))
        auxiliary = torch.load(auxiliary_dirs[1]/'last.pth', weights_only=True, map_location='cpu')
        self.assertEqual(outputs[0]['initial_state_sha256'], auxiliary['initial_state_sha256'])
        self.assertEqual(outputs[0]['exposure_chain'], auxiliary['exposure_chain'])
        self.assert_nested_equal(outputs[0]['torch_rng'], auxiliary['torch_rng'])
        self.assertFalse(torch.equal(outputs[0]['state_dict']['lstm.weight_ih_l0'], auxiliary['state_dict']['lstm.weight_ih_l0']))

    def test_observed_soft_target_loss_and_invalid_masks(self):
        logits=torch.zeros((2,12),requires_grad=True)
        targets=torch.from_numpy(np.stack([observed_tag_distribution([TAG_ORDER[0]]),observed_tag_distribution([TAG_ORDER[2],TAG_ORDER[4]])]))
        actual=tag_loss(logits,targets)
        self.assertAlmostEqual(float(actual.detach()),float(np.log(12)),places=6)
        actual.backward()
        self.assertAlmostEqual(float(logits.grad[1,2]),(.0833333333333333-.5)/2,places=6)
        for bad in (torch.zeros_like(targets),-targets,targets[:,None,:]):
            with self.assertRaises(ValueError):tag_loss(logits,bad)

    def test_artist_group_sampling_not_proportional_to_track_count(self):
        clips=clips_fixture()
        # Six tracks in one artist group and one in another: groups still receive equal probability.
        clips=[dict(row,group_id='large' if i<6 else 'small') for i,row in enumerate(clips)]
        sampler=PairedSampler(songs_fixture()['fit'],clips,dict(SETUP,aux_batch_size=100))
        counts={'large':0,'small':0}
        for step in range(1,21):
            for i in sampler.indices(step)[1]:counts[clips[i]['group_id']]+=1
        self.assertTrue(850<counts['small']<1150,counts)
        changed=[dict(row,group_id='different') if i==0 else row for i,row in enumerate(clips)]
        with self.assertRaisesRegex(ValueError,'membership'):
            PairedSampler(songs_fixture()['fit'],changed,dict(SETUP,aux_batch_size=100)).verify(sampler.state(20),20)

    def test_status_requires_real_lease_not_stale_running_json(self):
        rid='jamendo-test-status-fixture'
        folder,models=self.folders('status')
        atomic_json(folder/'config.json',dict(stage='D'))
        atomic_json(folder/'process.json',dict(process_identity=process_identity()))
        atomic_json(folder/'status.json',dict(stage='training',step=17))
        with patch.object(status_module,'paths',return_value=(folder,models)):
            self.assertEqual(status_module.snapshot(rid)['outcome'],'stopped_without_completion')
            with run_lease(rid):self.assertEqual(status_module.snapshot(rid)['outcome'],'running')

    def real_feature_entry(self):
        row=read_json(SOURCE)[0];protocol=read_json(PROTOCOL)
        entry=read_cached(row,CACHE,protocol['feature_key'])
        self.assertIsNotNone(entry)
        entry.update(role='auxiliary_fit',group_id=row['group_id'],observed_tags=row['selected_positive_tags'],
            tag_order=list(TAG_ORDER),observed_distribution=observed_tag_distribution(row['selected_positive_tags']).tolist(),
            audio_path=row['audio']['audio_path'],duration_seconds=row['audio']['native_audio']['duration_seconds'],
            local_time_origin_seconds=0,source_song_offset_seconds=0)
        return row,protocol,entry

    def test_real_feature_role_clock_hash_and_label_boundaries(self):
        row,protocol,entry=self.real_feature_entry()
        result=read_auxiliary_feature(entry,row,protocol['feature_key'])
        self.assertEqual(result['x'].shape[1],531)
        self.assertNotIn('y',result);self.assertNotIn('mask',result)
        for change in (dict(role='test'),dict(source_song_offset_seconds=1),dict(sha256='0'*64),
                       dict(observed_distribution=[0.]*12),dict(y=[[.1,.2]]),dict(group_id='different')):
            with self.assertRaises(ValueError):read_auxiliary_feature(dict(entry,**change),row,protocol['feature_key'])

    def test_real_pool_and_main_audio_overlap_gate(self):
        rows=read_json(SOURCE);protocol=read_json(PROTOCOL)
        validate_pool(rows,protocol,set())
        with self.assertRaisesRegex(ValueError,'overlap'):
            validate_pool(rows,protocol,{rows[0]['expected_audio_sha256']})
        altered=copy.deepcopy(rows);altered[0]['official_role']='validation'
        with self.assertRaises(ValueError):validate_pool(altered,protocol,set())


if __name__=='__main__':
    unittest.main(verbosity=2)
