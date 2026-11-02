"""Status is derived from a matched record chain, never a model's display name."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from tools.train_next_v2.v3_workbench.studio import store, prediction_status as status
from tools.train_next_v2.v3_workbench.studio.paths import WORKSPACE


class PredictionStatusTests(unittest.TestCase):
    def setUp(self):
        base = WORKSPACE/'data/v3.0/test-workbench'
        base.mkdir(exist_ok=True, parents=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        audio = root/'audio.wav'
        with wave.open(str(audio), 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000)
            f.writeframes(bytes(16000))
        self.manifest = root/'manifest.json'
        manifest = dict(schema_version=3, status='research_preview_not_promoted', promoted_model=False,
                        systems={key:dict(calibration='unavailable', identity=key) for key in ('b01','b00')},
                        legacy=dict(identity='legacy'), calibration=dict(status='unavailable'))
        store.atomic_json(self.manifest, manifest)
        mh = store.digest(self.manifest)
        systems = {key:dict(values=[[.1,.2],[.3,.4]], times_us=[0,500000], step_us=500000,
                           model_path=str(self.manifest), model_sha256=store.object_digest(
                               manifest['legacy'] if key=='v2' else manifest['systems'][key]))
                   for key in ('b01','b00','v2')}
        self.work = store.create_project(root/'works','status contract',audio,None,None,
                                        dict(duration=1.,tempo=0,beat_times=[]),systems=systems)
        audio_meta = dict(sha256=store.digest(audio), duration_seconds=1.)
        source = dict(schema_version=3,id='source',audio=audio_meta,manifest_sha256=mh)
        result = dict(schema_version=3,project_id='source',audio=audio_meta,manifest_sha256=mh,
                      provenance=dict(promoted_model=False),calibration=dict(status='unavailable',values=None),
                      systems={k:dict(raw_va=[[.1,.2],[.3,.4]],times_seconds=[0,.5],calibrated=False) for k in systems})
        archive = self.work/'analysis/source-v3'
        store.atomic_json(archive/'project.json',source)
        store.atomic_json(archive/'analysis/result.json',result)
        receipt = dict(schema_version=3,manifest_sha256=mh,audio_sha256=audio_meta['sha256'],
                       project_sha256=store.digest(archive/'project.json'),result_sha256=store.digest(archive/'analysis/result.json'))
        store.atomic_json(archive/'analysis/receipt.json',receipt)
        self.meta = store.read_json(self.work/'project.json')
        self.meta['provenance'] = dict(kind='frozen-v3-analysis',manifest_sha256=mh,
            source_project_sha256=receipt['project_sha256'],source_result_sha256=receipt['result_sha256'],
            source_receipt_sha256=store.digest(archive/'analysis/receipt.json'))
        store.atomic_json(self.work/'project.json',self.meta)
        self.override = patch.object(status,'MANIFEST',self.manifest)
        self.override.start(); self.addCleanup(self.override.stop)

    def rows(self):
        return {row['id']:row['prediction_status'] for row in store.systems_metadata(self.work)}

    def test_verified_research_and_legacy_remain_separate_and_read_only(self):
        before = {str(p):store.digest(p) for p in self.work.rglob('*') if p.is_file()}
        rows = self.rows()
        for key in ('b01','b00'):
            self.assertEqual(rows[key]['research'],'not_promoted')
            self.assertEqual(rows[key]['calibration'],'unavailable')
        self.assertEqual(rows['v2']['research'],'not_recorded')
        self.assertEqual(rows['v2']['calibration'],'not_applied')
        self.assertEqual(before,{str(p):store.digest(p) for p in self.work.rglob('*') if p.is_file()})

    def test_no_evidence_and_unknown_name_never_claim_verified(self):
        for provenance in ({},None,{'kind':'synthetic'}):
            meta=copy.deepcopy(self.meta);meta['provenance']=provenance
            rows=status.prediction_statuses(self.work,meta,{'v2':{},'b01':{},'future':{}})
            self.assertTrue(all(r['evidence']=='unverified' for r in rows.values()))
            self.assertEqual(rows['v2']['kind'],'legacy')
            self.assertEqual(rows['future']['kind'],'unknown')

    def test_comparison_arrays_can_skip_redundant_status_work(self):
        with patch.object(status,'prediction_statuses',side_effect=AssertionError('duplicate status work')):
            rows=store.systems_metadata(self.work,include_values=True,include_status=False)
        self.assertEqual(len(rows),3)
        self.assertNotIn('prediction_status',rows[0])
        self.assertEqual(rows[0]['times_us'],[0,500000])
        self.assertEqual(rows[0]['values'],[[.1,.2],[.3,.4]])

    def test_changed_missing_or_invalid_manifest_has_no_stale_success(self):
        original=self.manifest.read_bytes()
        for data in (b'{}',b'not json'):
            self.manifest.write_bytes(data)
            self.assertTrue(all(r['evidence']=='unverified' for r in self.rows().values()))
            self.manifest.write_bytes(original)
            self.assertEqual(self.rows()['b01']['evidence'],'verified_frozen_record')
        self.manifest.unlink()
        self.assertTrue(all(r['evidence']=='unverified' for r in self.rows().values()))

    def test_model_digest_and_path_mismatch_are_per_system(self):
        for field,value in [('model_sha256','wrong'),('model_path',str(self.manifest.parent/'other.json'))]:
            records=copy.deepcopy(self.meta['systems']);records['b01'][field]=value
            rows=status.prediction_statuses(self.work,self.meta,records)
            self.assertEqual(rows['b01']['evidence'],'unverified')
            self.assertEqual(rows['b00']['evidence'],'verified_frozen_record')

    def test_archive_or_audio_identity_mismatch_fails_closed(self):
        meta=copy.deepcopy(self.meta);meta['audio_sha256']='changed'
        rows=status.prediction_statuses(self.work,meta,meta['systems'])
        self.assertTrue(all(r['evidence']=='unverified' for r in rows.values()))
        file=self.work/'analysis/source-v3/analysis/receipt.json'
        file.write_text('{}')
        self.assertTrue(all(r['evidence']=='unverified' for r in self.rows().values()))

    def test_changed_clock_mask_or_values_do_not_inherit_evidence(self):
        file=self.work/self.meta['systems']['b01']['file']
        for field,value in [('times_us',[0,400000]),('mask',[[True,False],[True,True]]),('values',[[.1,.9],[.3,.4]])]:
            data=dict(values=[[.1,.2],[.3,.4]],times_us=[0,500000],mask=[[True,True],[True,True]])
            data[field]=value;np.savez(file,**data)
            meta=copy.deepcopy(self.meta);meta['systems']['b01']['raw_sha256']=store.digest(file)
            meta['systems_sha256']=store.object_digest(meta['systems']);store.atomic_json(self.work/'project.json',meta)
            self.assertEqual(self.rows()['b01']['evidence'],'unverified')
            self.assertEqual(self.rows()['b00']['evidence'],'verified_frozen_record')

    def test_corrupt_raw_file_degrades_only_its_status(self):
        file=self.work/self.meta['systems']['b01']['file']
        file.write_bytes(b'PK\x03\x04broken')
        meta=copy.deepcopy(self.meta);meta['systems']['b01']['raw_sha256']=store.digest(file)
        meta['systems_sha256']=store.object_digest(meta['systems']);store.atomic_json(self.work/'project.json',meta)
        self.assertEqual(self.rows()['b01']['evidence'],'unverified')
        self.assertEqual(self.rows()['b00']['evidence'],'verified_frozen_record')

    def test_extreme_duration_in_bound_record_degrades_safely(self):
        archive=self.work/'analysis/source-v3'
        result=store.read_json(archive/'analysis/result.json');result['audio']['duration_seconds']=1e308
        source=store.read_json(archive/'project.json');source['audio']=result['audio']
        store.atomic_json(archive/'project.json',source);store.atomic_json(archive/'analysis/result.json',result)
        receipt=store.read_json(archive/'analysis/receipt.json')
        receipt.update(project_sha256=store.digest(archive/'project.json'),result_sha256=store.digest(archive/'analysis/result.json'))
        store.atomic_json(archive/'analysis/receipt.json',receipt)
        meta=copy.deepcopy(self.meta);meta['provenance'].update(source_project_sha256=receipt['project_sha256'],
            source_result_sha256=receipt['result_sha256'],source_receipt_sha256=store.digest(archive/'analysis/receipt.json'))
        store.atomic_json(self.work/'project.json',meta)
        self.assertTrue(all(r['evidence']=='unverified' for r in self.rows().values()))


if __name__=='__main__':
    unittest.main()
