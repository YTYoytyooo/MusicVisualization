import test_annotation as legacy
from server import ContinuousStore, Conflict, overwrite_tracks
import copy
import csv
import io
import unittest

class ContinuousTests(unittest.TestCase):
    # Inherit setup only; legacy contracts remain tested by AnnotationTests.
    def setUp(self):
        legacy.AnnotationTests.setUp(self)
        self.store.close()
        self.store=ContinuousStore(self.audio,self.root/'continuous')

    tearDown = legacy.AnnotationTests.tearDown

    def body(self):
        doc=self.store.load('alice',self.key)
        return dict(base_revision=doc['revision'],audio_sha256=doc['source']['audio_sha256'],duration=10,
                    takes=[dict(id='take1',axes=['valence'],points=[dict(time=.2,status='annotated',valence=0,arousal=None,confidence='high'),
                          dict(time=1.25,status='uncertain',valence=None,arousal=None,confidence='')])],
                    transitions=[dict(id='marker1',time=.75,kind='sudden',note='鼓点进入')])

    def test_continuous_persistence_and_markers(self):
        body=self.body();result=self.store.save('alice',self.key,body)
        self.assertNotIn('segments',result)
        self.assertEqual(result['takes'],body['takes'])
        self.assertEqual(result['transitions'],body['transitions'])
        with self.assertRaises(Conflict):self.store.save('alice',self.key,body)
        body['base_revision']=1;body['transitions'][0]['time']=.9
        self.store.save('alice',self.key,body)
        self.assertTrue((self.root/'continuous/alice/history'/self.key/'r000001.json').exists())
        self.assertEqual(self.store.load('alice',self.key)['takes'],result['takes'])

    def test_invalid_continuous_points(self):
        for patch in ({'time':-.1},{'time':11},{'valence':True},{'valence':None},{'confidence':'invalid'}):
            body=self.body();body['takes'][0]['points'][0].update(patch)
            with self.assertRaises(ValueError):self.store.save('alice',self.key,body)
        body=self.body();body['takes'][0]['points'][1]['time']=.1
        with self.assertRaises(ValueError):self.store.save('alice',self.key,body)
        body=self.body();body['transitions'][0]['time']=11
        with self.assertRaises(ValueError):self.store.save('alice',self.key,body)

    def test_export_preserves_transition_and_uncertainty(self):
        self.store.save('alice',self.key,self.body())
        rows=list(csv.DictReader(io.StringIO(self.store.export('alice','valid').decode('utf-8-sig'))))
        self.assertEqual([r['record_type'] for r in rows],['sample','transition'])
        self.assertEqual(rows[0]['valence'],'0.0');self.assertEqual(rows[1]['time_seconds'],'0.75')
        rows=list(csv.DictReader(io.StringIO(self.store.export('alice','all').decode('utf-8-sig'))))
        self.assertEqual(rows[1]['status'],'uncertain');self.assertEqual(rows[1]['valence'],'')

    def test_repeated_takes_replace_previous(self):
        body=self.body()
        repeated=copy.deepcopy(body['takes'][0]);repeated['id']='take2'
        repeated['points'][0]['valence']=-.5
        body['takes'].append(repeated)
        saved=self.store.save('alice',self.key,body)
        self.assertEqual(len(saved['takes']),1)
        self.assertEqual(saved['takes'][0]['points'][0]['valence'],-.5)

    def test_legacy_files_are_not_reinterpreted_or_overwritten(self):
        import json
        path=self.store.file('alice',self.key)
        path.parent.mkdir(parents=True,exist_ok=True)
        old={'revision':1,'protocol':'perceived-musical-expression-va-segment-v1',
             'source':self.store.source(self.key),'segments':[]}
        path.write_text(json.dumps(old),encoding='utf-8')
        original=path.read_bytes()
        with self.assertRaises(Conflict):self.store.load('alice',self.key)
        with self.assertRaises(Conflict):self.store.save('alice',self.key,{})
        self.assertEqual(path.read_bytes(),original)

    def test_audio_change_rejected(self):
        self.store.save('alice',self.key,self.body())
        with self.path.open('ab') as f:f.write(b'changed')
        with self.assertRaises(Conflict):self.store.load('alice',self.key)
        with self.assertRaises(Conflict):self.store.export('alice','json')

    def test_partial_axis_overwrite_preserves_other_axis_and_outside(self):
        body=self.body()
        body['takes']=[dict(id='old',axes=['valence','arousal'],points=[
            dict(time=t,status='annotated',valence=0,arousal=.5,confidence='high') for t in (0,1,2,3,4)])]
        initial=self.store.save('alice',self.key,body)
        body['base_revision']=1
        body['takes']=initial['takes']+[dict(id='new',axes=['valence'],points=[
            dict(time=t,status='annotated',valence=-.5,arousal=None,confidence='medium') for t in (1,3)])]
        saved=self.store.save('alice',self.key,body)
        values={axis:sorted((p['time'],p[axis]) for take in saved['takes'] if take['axes']==[axis] for p in take['points']) for axis in ('valence','arousal')}
        self.assertEqual(values['valence'],[(0,0),(1,-.5),(3,-.5),(4,0)])
        self.assertEqual(values['arousal'],[(t,.5) for t in (0,1,2,3,4)])
        self.assertEqual(overwrite_tracks(saved['takes']),saved['takes'])
        self.assertEqual(saved['transitions'],initial['transitions'])

    def test_uncertainty_overwrites_old_values_and_unselected_axis_is_rejected(self):
        body=self.body()
        body['takes'].append(dict(id='uncertain',axes=['valence'],points=[
            dict(time=.2,status='uncertain',valence=None,arousal=None,confidence='')]))
        saved=self.store.save('alice',self.key,body)
        self.assertTrue(all(p['valence'] is None for take in saved['takes'] for p in take['points']))
        bad=self.body();bad['takes'][0]['points'][0]['arousal']=.5
        with self.assertRaises(ValueError):self.store.save('alice',self.key,bad)

    def test_optional_confidence_can_be_updated_later(self):
        body=self.body();body['takes'][0]['points'][0]['confidence']=''
        first=self.store.save('alice',self.key,body)
        rows=list(csv.DictReader(io.StringIO(self.store.export('alice','valid').decode('utf-8-sig'))))
        self.assertEqual([r['record_type'] for r in rows],['transition'])
        self.assertEqual(self.store.catalog('alice')[0]['unrated'],1)
        body['base_revision']=1;body['takes'][0]['points'][0]['confidence']='medium'
        second=self.store.save('alice',self.key,body)
        self.assertEqual(first['takes'][0]['points'][0]['valence'],second['takes'][0]['points'][0]['valence'])
        self.assertEqual(self.store.catalog('alice')[0]['unrated'],0)

    def test_completion_is_per_reviewer_and_per_axis(self):
        body=self.body();body['completed']={'valence':True,'arousal':False}
        self.store.save('alice',self.key,body)
        self.assertTrue(self.store.catalog('alice')[0]['completed']['valence'])
        self.assertFalse(self.store.catalog('bob')[0]['completed']['valence'])
        body['base_revision']=1;body['completed']['arousal']=True
        with self.assertRaises(ValueError):self.store.save('alice',self.key,body)

    def test_transition_type_is_optional_and_legacy_type_is_preserved(self):
        body=self.body();body['transitions'].append(dict(id='plain',time=2,note='变化'))
        saved=self.store.save('alice',self.key,body)
        self.assertEqual(saved['transitions'][0]['kind'],'sudden')
        self.assertNotIn('kind',saved['transitions'][1])
        rows=list(csv.DictReader(io.StringIO(self.store.export('alice','all').decode('utf-8-sig'))))
        plain=next(row for row in rows if row['record_type']=='transition' and row['time_seconds']=='2.0')
        self.assertEqual(plain['transition_kind'],'')

    def test_import_is_local_non_destructive_and_rejects_invalid_uploads(self):
        data=self.path.read_bytes()
        imported=self.store.import_audio('新音乐.wav',io.BytesIO(data),len(data))
        again=self.store.import_audio('新音乐.wav',io.BytesIO(data),len(data))
        self.assertEqual(imported,again)
        self.assertEqual(len(self.store.files),2)
        self.assertEqual(self.store.files[imported['id']][0].read_bytes(),data)
        self.assertEqual(self.path.read_bytes(),data)
        for name,content,length in [('../out.wav',data,len(data)),('script.exe',data,len(data)),
                                    ('fake.wav',b'not audio',9),('cut.wav',data,len(data)+1),('big.wav',data,513*1024*1024)]:
            with self.assertRaises(ValueError):self.store.import_audio(name,io.BytesIO(content),length)
        self.assertEqual(len(self.store.files),2)
        self.assertEqual(list((self.audio/'imports').glob('*.part')),[])

if __name__=='__main__':unittest.main()
