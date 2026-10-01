"""Persistence and HTTP contract checks using disposable synthetic audio."""
from pathlib import Path
import copy
import csv
import http.client
import io
import json
import struct
import sys
import tempfile
import threading
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import Store, Conflict, make_server


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio = self.root/'audio'
        self.audio.mkdir()
        self.path = self.audio/'example.wav'
        with wave.open(str(self.path), 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000)
            f.writeframes(struct.pack('<h', 1)*80000)
        self.store = Store(self.audio, self.root/'labels')
        self.key = next(iter(self.store.files))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def body(self):
        doc = self.store.load('alice',self.key)
        return {'base_revision':doc['revision'], 'audio_sha256':doc['source']['audio_sha256'],
                'duration':10, 'artist':'', 'style':'', 'segments':[
                    {'id':'a','start':0,'end':5,'status':'annotated','valence':0,'arousal':-.5,
                     'confidence':'high','vocals':'instrumental','note':''},
                    {'id':'b','start':5,'end':10,'status':'uncertain','valence':None,'arousal':None,'note':'难以判断'}]}

    def test_save_reload_revision_conflict_and_history(self):
        body=self.body()
        first=self.store.save('alice',self.key,body)
        self.assertEqual(first['segments'][0]['valence'],0)
        with self.assertRaises(Conflict): self.store.save('alice',self.key,body)
        body['base_revision']=1; body['segments'][0]['valence']=.5
        self.store.save('alice',self.key,body)
        self.assertEqual(self.store.load('alice',self.key)['revision'],2)
        history=json.loads((self.root/'labels/alice/history'/self.key/'r000001.json').read_text())
        self.assertEqual(history['segments'][0]['valence'],0)

    def test_invalid_or_missing_ratings_never_saved(self):
        for bad in (None, True, float('nan'), 1.1, '0.5'):
            body=self.body();body['segments'][0]['valence']=bad
            with self.assertRaises(ValueError): self.store.save('alice',self.key,body)
        self.assertFalse(self.store.file('alice',self.key).exists())

    def test_overlap_bounds_and_duplicate_ids_rejected(self):
        for patch in ({'start':4}, {'end':12}, {'id':'a'}, {'start':10}):
            body=self.body();body['segments'][1].update(patch)
            with self.assertRaises(ValueError): self.store.save('alice',self.key,body)

    def test_uncertain_is_not_neutral_and_low_confidence_excluded(self):
        body=self.body();self.store.save('alice',self.key,body)
        rows=list(csv.DictReader(io.StringIO(self.store.export('alice','valid').decode('utf-8-sig'))))
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['valence'],'0.0')
        self.assertEqual(rows[0]['label_scope'],'segment_mean')
        all_rows=list(csv.DictReader(io.StringIO(self.store.export('alice','all').decode('utf-8-sig'))))
        self.assertEqual(all_rows[1]['valence'],'')
        body['base_revision']=1;body['segments'][0]['confidence']='low'
        self.store.save('alice',self.key,body)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(self.store.export('alice','valid').decode('utf-8-sig'))))),0)

    def test_reviewers_isolated_and_path_traversal_rejected(self):
        self.store.save('alice',self.key,self.body())
        self.assertEqual(self.store.load('bob',self.key)['segments'],[])
        for reviewer in ('../alice', 'CON', '', 'a/b'):
            with self.assertRaises(ValueError):self.store.load(reviewer,self.key)

    def test_audio_change_blocks_old_labels_and_export(self):
        self.store.save('alice',self.key,self.body())
        with self.path.open('ab') as f:f.write(b'changed')
        with self.assertRaises(Conflict):self.store.load('alice',self.key)
        with self.assertRaises(Conflict):self.store.export('alice','valid')

    def test_output_lock_and_csv_formula_escaping(self):
        with self.assertRaises(ValueError):Store(self.audio,self.root/'labels')
        body=self.body();body['segments'][0]['note']='=1+1'
        self.store.save('alice',self.key,body)
        rows=list(csv.DictReader(io.StringIO(self.store.export('alice','all').decode('utf-8-sig'))))
        self.assertEqual(rows[0]['note'],"'=1+1")
        self.assertEqual(self.store.load('alice',self.key)['segments'][0]['note'],'=1+1')

    def test_http_token_and_range(self):
        httpd=make_server(self.audio,self.root/'http-labels',0,mode='segments')
        thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
        try:
            client=http.client.HTTPConnection('127.0.0.1',httpd.server_port,timeout=5)
            client.request('GET','/api/library?reviewer=alice');response=client.getresponse()
            catalog=json.loads(response.read());self.assertEqual(response.status,200)
            client.request('GET',f'/audio/{self.key}',headers={'Range':'bytes=0-15'})
            response=client.getresponse();self.assertEqual(response.status,206);self.assertEqual(len(response.read()),16)
            client.request('GET',f'/audio/{self.key}',headers={'Range':'bytes=99999999-'})
            response=client.getresponse();self.assertEqual(response.status,416);response.read()
            body=json.dumps({**self.body(),'reviewer':'alice','song_id':self.key})
            client.request('POST','/api/save',body,{'Content-Type':'application/json'})
            response=client.getresponse();self.assertEqual(response.status,403);response.read()
            client.request('POST','/api/save',body,{'Content-Type':'application/json','X-Annotation-Token':catalog['token']})
            response=client.getresponse();self.assertEqual(response.status,200);response.read()
            client.close()
        finally:
            httpd.shutdown();httpd.server_close();httpd.store.close();thread.join()


if __name__=='__main__':unittest.main()
