"""Render fixed examples without tuning, smoothing, interpolation or label shifts."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import html
import json
import os
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import is_running, lock_path, run_lease
from tools.train_next_v2.common import ROOT, BASE, PLAN, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.research_inference.benchmark import process_alive
from tools.train_next_v2.run import Tee
from tools.train_next_v2.cross_version_eval.job import DIRECTORY as SOURCE, PROTOCOL, key_for, stores
from tools.train_next_v2.cross_version_eval.methods import AXES, SYSTEMS

JOB = 'f-v2-mixed-b01-curves-v1'
DIRECTORY = BASE / 'comparison_reports' / JOB
VENDOR = BASE / 'resources/plotting-cp313-v1'
COLORS = ('#2878b5', '#dc7633', '#9a7fc1', '#9a7fc1', '#9a7fc1', '#6d3dad')
LABELS = ('V2', 'Mixed', 'B01 seed 03', 'B01 seed 04', 'B01 seed 05', 'B01 ensemble')
DOMAINS = {'deam': 'DEAM', 'memo2496': 'Memo2496', 'pmemo': 'PMEmo'}


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def number(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def segments(times, native):
    times = np.asarray(times)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid observed clock')
    return [part for part in np.split(np.arange(len(times)), np.flatnonzero(np.diff(times) > native + 1e-7) + 1) if len(part)]


def check_segments():
    assert [p.tolist() for p in segments([0., 1., 3., 4.], 1.)] == [[0, 1], [2, 3]]
    assert [p.tolist() for p in segments([.5, 1., 2.], .5)] == [[0, 1], [2]]
    assert segments([], .5) == []
    return 'One-second native labels connect; genuinely missing intervals remain disconnected.'


def checked_inputs():
    protocol, report, verification = [read_json(p) for p in (PROTOCOL, SOURCE/'report.json', SOURCE/'verification.json')]
    if (verification['status'] != 'passed' or verification['verified_songs'] != 584
            or verification['report_sha256'] != sha256(SOURCE/'report.json')
            or verification['protocol_sha256'] != sha256(PROTOCOL)
            or verification['launch_sha256'] != sha256(SOURCE/'launch.json')):
        raise ValueError('Unverified comparison input')
    reps = protocol['representatives']
    if len(reps) != 18 or len({r['song_id'] for r in reps}) != 18:
        raise ValueError('Frozen representative selection changed')
    if report['final_test_songs_used'] != 0 or report['promoted_model'] is not False:
        raise ValueError('Unexpected evaluation scope')
    return protocol, report, verification


def state(stage, **values):
    row = dict(job_id=JOB, stage=stage, updated_at=now(), **values)
    atomic_json(DIRECTORY/'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def setup_plotting():
    if not os.environ.get('MPLCONFIGDIR'):
        os.environ['MPLCONFIGDIR'] = str(BASE/'cache/matplotlib-comparison-report')
    sys.path.append(str(VENDOR))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import PIL
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'axes.grid': True,
                         'grid.alpha': .15, 'figure.facecolor': 'white'})
    return plt, dict(python=sys.version, numpy=np.__version__, matplotlib=matplotlib.__version__, pillow=PIL.__version__)


def check():
    directory=BASE/'analyses/f-comparison-report-smoke-v1'
    if directory.exists(): raise ValueError('Preserve prior smoke-check evidence')
    directory.mkdir(parents=True)
    code={rel(p):sha256(p) for p in Path(__file__).parent.glob('*.py')}
    registration=dict(at=now(),original_argv=[sys.executable,'-u','-m',__package__+'.job','--check'],
                      cwd=str(ROOT),code=code,environment={'MPLCONFIGDIR':os.environ.get('MPLCONFIGDIR')},
                      source_report_sha256=sha256(SOURCE/'report.json'))
    atomic_json(directory/'launch.json',registration)
    record_event(dict(event='F_comparison_renderer_smoke_registered',registration=rel(directory/'launch.json')))
    protocol,report,_=checked_inputs(); plt,versions=setup_plotting(); _,outputs=stores()
    selection=protocol['representatives'][0]
    index,row=next((i,r) for i,r in enumerate(protocol['validation_songs']) if r['song_id']==selection['song_id'])
    _,saved=outputs.read(key_for(index,row),row['song_id'])
    scores=next(r['scores'] for r in report['songs'] if r['song_id']==row['song_id'])
    fig=figure(plt,curve_payload(selection,row,saved,scores)); path=directory/'preview.png'
    fig.savefig(path,dpi=140); plt.close(fig)
    from PIL import Image
    with Image.open(path) as im: assert im.size==(2240,1680); im.verify()
    result=dict(status='passed',at=now(),versions=versions,native_gap_check=check_segments(),
                preview=rel(path),preview_sha256=sha256(path),source_code=code,song_id=row['song_id'])
    atomic_json(directory/'report.json',result)
    record_event(dict(event='F_comparison_renderer_smoke_passed',report=rel(directory/'report.json')))
    print(json.dumps(result))


def prepare():
    if (DIRECTORY/'launch.json').exists():
        raise ValueError('Preserve registered renderer configuration')
    protocol, report, verification = checked_inputs()
    source_launch = read_json(SOURCE/'launch.json')
    smoke_path=BASE/'analyses/f-comparison-report-smoke-v1/report.json'
    smoke=read_json(smoke_path)
    if smoke['status']!='passed': raise ValueError('Renderer smoke check required')
    verify_hashes(smoke['source_code'])
    verify_hashes(source_launch['pinned_hashes'])
    vendor = read_json(VENDOR/'provenance.json')
    pins = {rel(VENDOR/name): digest for name, digest in vendor['file_sha256'].items()}
    paths = [PROTOCOL, SOURCE/'report.json', SOURCE/'verification.json', SOURCE/'launch.json', VENDOR/'provenance.json',smoke_path]
    paths += list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/name for name in ('tools/train_next_v2/common.py', 'tools/train_next/core.py',
               'tools/train_next/runtime.py', 'tools/train_next_v2/run.py',
               'tools/train_next_v2/research_inference/benchmark.py', 'tools/train_next_v2/cross_version_eval/storage.py')]
    plan = read_json(PLAN)
    paths += [ROOT/p for p in plan['stages']['F']['runtime_interruption_notes']]
    for path in paths:
        pins[rel(path)] = sha256(path)
    _, outputs = stores()
    for selection in protocol['representatives']:
        index, row = next((i, r) for i, r in enumerate(protocol['validation_songs']) if r['song_id'] == selection['song_id'])
        key = key_for(index, row)
        receipt, _ = outputs.read(key, row['song_id'])
        for path in outputs.paths(key)[:2]:
            pins[rel(path)] = sha256(path)
    env = {**source_launch['environment'], 'MPLCONFIGDIR': str(BASE/'cache/matplotlib-comparison-report')}
    argv = [source_launch['python'], '-u', '-m', __package__+'.job', '--launch', str(DIRECTORY/'launch.json')]
    launch = dict(job_id=JOB, registered_at=now(), python=source_launch['python'], cwd=str(ROOT), environment=env,
                  original_argv=argv, resume_argv=argv+['--resume'], verification_argv=argv+['--verify'],
                  source_report=rel(SOURCE/'report.json'), source_verification=rel(SOURCE/'verification.json'),
                  pinned_hashes=pins, source_launch_sha256=sha256(SOURCE/'launch.json'),
                  representatives=protocol['representatives'], log=rel(DIRECTORY/'console.log'),
                  os_lock=str(lock_path(JOB)), status_module=__package__+'.job --status',
                  native_gap_check=check_segments(), runtime_notes=plan['stages']['F']['runtime_interruption_notes'],
                  acceptance='18 frozen songs, per-axis exact common clocks, 18 PNG/SVG/raw JSON receipts, readable offline HTML, source metrics unchanged; no calibration/smoothing/lag correction.',
                  recovery='Only after confirming exact PID exit and free OS lease, diagnose failure then use original argv plus --resume. Reuse only checksum-valid committed figure groups. Preserve uncommitted files for investigation; never overwrite configuration or passed verification.')
    atomic_json(DIRECTORY/'launch.json', launch)
    record_event(dict(event='F_comparison_renderer_registered',job_id=JOB,launch=rel(DIRECTORY/'launch.json')),
                 run=dict(run_id=JOB,stage='F',status='registered',launch=rel(DIRECTORY/'launch.json')))
    print(json.dumps(dict(job_id=JOB,launch=rel(DIRECTORY/'launch.json'),pinned_inputs=len(pins))))


def curve_payload(selection, row, saved, scores):
    payload = dict(selection=selection, song_id=row['song_id'], duration_seconds=row['duration_seconds'],
                   native_seconds=float(saved['native_seconds']), axes={})
    for a, axis in enumerate(AXES):
        valid = saved['common_mask'][:,a]
        times = saved['times'][valid]
        payload['axes'][axis] = dict(times=times.tolist(), truth=saved['y'][valid,a].tolist(),
            predictions={s:saved['predictions'][i,valid,a].tolist() for i,s in enumerate(SYSTEMS)},
            segments=[p.tolist() for p in segments(times,payload['native_seconds'])],
            metrics={s:scores[s][axis] for s in SYSTEMS})
    return payload


def figure(plt, payload):
    fig = plt.figure(figsize=(16,12))
    grid = fig.add_gridspec(4,2,height_ratios=(2.4,1.6,1.6,1.3))
    fig.subplots_adjust(left=.065,right=.98,bottom=.065,top=.90,hspace=.58,wspace=.20)
    fig.suptitle(f"{payload['song_id']}  |  fixed example: {payload['selection']['category']}", x=.065, y=.975, ha='left', fontsize=18)
    fig.text(.065,.943,'Raw V/A on the exact shared label clock. Pale B01 lines = all three seeds; bold purple = secondary ensemble.',fontsize=10)
    for a,axis in enumerate(AXES):
        data=payload['axes'][axis]; t=np.asarray(data['times']); y=np.asarray(data['truth'])
        raw=fig.add_subplot(grid[0,a]); centered=fig.add_subplot(grid[1,a]); lag=fig.add_subplot(grid[2,a])
        parts=[np.asarray(part) for part in data['segments']]
        for i,part in enumerate(parts):
            raw.plot(t[part],y[part],color='#138878',lw=2,label='Truth' if i==0 else None)
            centered.plot(t[part],y[part]-y.mean(),color='#138878',lw=1.8)
        residual_limit=max(.1, float(np.max(np.abs(y-y.mean()),initial=0)))
        for s,system in enumerate(SYSTEMS):
            p=np.asarray(data['predictions'][system]); pale=s in (2,3,4)
            for i,part in enumerate(parts):
                style=dict(color=COLORS[s],lw=.7 if pale else 1.25,alpha=.40 if pale else .95)
                raw.plot(t[part],p[part],label=(LABELS[s] if not pale else 'B01 seeds' if s==2 else None) if i==0 else None,**style)
                centered.plot(t[part],p[part]-p.mean(),**style)
            residual_limit=max(residual_limit,float(np.max(np.abs(p-p.mean()),initial=0)))
            if s in (0,1,5):
                diagnostic=data['metrics'][system]['lag']
                if 'search_seconds' in diagnostic:
                    lag.plot(diagnostic['search_seconds'],diagnostic['correlations'],color=COLORS[s],lw=1.25,label=LABELS[s])
        raw.set(title=axis.capitalize()+' | raw values',ylim=(-1.05,1.05),xlim=(0,payload['duration_seconds']),ylabel='V/A [-1, 1]')
        raw.legend(loc='lower left',fontsize=7,ncol=3)
        centered.set(title='Changes after subtracting each curve mean (diagnostic)',ylim=(-residual_limit*1.12,residual_limit*1.12),xlim=(0,payload['duration_seconds']),xlabel='Original audio time (s)',ylabel='Centered V/A')
        lag.axvline(0,color='#888888',lw=.8,ls='--')
        lag.set(title='Lag diagnostic only; no correction applied',xlabel='Prediction lag (s); positive = follows truth',ylabel='Pearson r',ylim=(-1.05,1.05))
        lag.legend(fontsize=7,loc='lower right')
        table_axis=fig.add_subplot(grid[3,a]); table_axis.axis('off')
        m=data['metrics']; lines=[['Truth',number(float(y.mean()),3),number(float(y.std()),3),'—','—','—']]
        for system in ('v2','mixed','b01_mean','b01_ensemble'):
            if system=='b01_mean':
                values={k:float(np.mean([m[s][k] for s in SYSTEMS[2:5]])) if all(m[s][k] is not None for s in SYSTEMS[2:5]) else None for k in ('prediction_mean','prediction_std','mae','ccc')}
                name='B01 seed-metric mean'; best='—'
            else:
                values=m[system]; name={'v2':'V2','mixed':'Mixed','b01_ensemble':'B01 ensemble'}[system]
                diagnostic=values['lag']; best=number(diagnostic['best_seconds'],1)+(' *' if not diagnostic['credible'] else ' ?')
            lines.append([name]+[number(values[k],3) for k in ('prediction_mean','prediction_std','mae','ccc')]+[best])
        tab=table_axis.table(cellText=lines,colLabels=['Curve / metric','Mean','SD','MAE','CCC','Lag(s)'],cellLoc='center',loc='center',colWidths=[.34,.13,.13,.13,.13,.14])
        tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1,1.25)
        for (r,c),cell in tab.get_celld().items():
            cell.set_edgecolor('#d9e0e6'); cell.set_facecolor('#eef3f8' if r==0 else '#ffffff')
    fig.text(.065,.025,'Blank spans have no shared label support. Lag * = unreliable; ? = diagnostic candidate requiring human curve review. No smoothing, calibration or label shifts.',fontsize=9)
    return fig


def table(headers, rows):
    cell=lambda v: html.escape(str(v))
    return '<table><thead><tr>'+''.join('<th>'+cell(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def document(protocol, report, launch, receipts):
    ag=report['aggregate']; systems=ag['systems']; esc=html.escape
    chosen=('v2','mixed','b01_mean','b01_ensemble')
    names={'v2':'V2 原版推理','mixed':'已完成 mixed','b01_mean':'B01 三种子指标均值（主结果）','b01_ensemble':'B01 固定曲线均值（次结果）'}
    rows=[[names[s]]+[number(systems[s]['overall'][m]) for m in ('ccc','mae','rmse','mean_absolute_bias','centered_rmse','derivative_mae')]+[f"{systems[s]['overall']['dynamics_beats_constant']:.1%}"] for s in chosen]
    sections=['<h1>连续情绪模型：584 首跨版本比较</h1><p>独立核验通过；研究比较，尚未达到发布门槛。</p>',
        '<p>逐曲计算，再按三个数据库 × V/A 六个单元等权汇总；原始取值为 [-1,1]。B01 主结果为三个种子各自指标的均值，固定集成曲线只作为次结果。CCC 越大越好，其余误差越小越好。</p>',
        table(['模型','CCC','MAE','RMSE','歌曲均值绝对偏差','动态残差 RMSE','变化率 MAE','动态优于自身均值常量'],rows),
        '<h2>逐库逐维结果</h2>']
    rows=[]
    for cell in systems['v2']['cells']:
        d,a=cell.split('/')
        row=[DOMAINS[d]+' / '+('效价 V' if a=='valence' else '唤醒 A')]
        for m in ('ccc','mae','rmse'):
            row.extend(number(systems[s]['cells'][cell][m]) for s in ('v2','mixed','b01_mean'))
        rows.append(row)
    sections.append(table(['单元','CCC V2','CCC mixed','CCC B01','MAE V2','MAE mixed','MAE B01','RMSE V2','RMSE mixed','RMSE B01'],rows))
    sections.append('<h2>配对歌曲组 bootstrap（2,000 次，95% 区间）</h2>')
    rows=[]
    for diff in ('b01_mean_minus_v2','b01_mean_minus_mixed'):
        values=ag['paired_differences'][diff]
        for cell,metrics in [('overall',values['overall'])]+list(values['cells'].items()):
            ci=metrics['ccc']; rows.append([diff,cell,number(ci['point']),number(ci['lower'])+' ～ '+number(ci['upper'])])
    sections.append(table(['比较','单元','CCC 差值','95% 区间'],rows))
    sections.append('<h2>三种子稳定性</h2>')
    rows=[[s]+[number(systems[s]['overall'][m]) for m in ('ccc','mae','rmse')] for s in SYSTEMS[2:5]]
    rows.append(['种子样本标准差']+[number(ag['b01_seed_sample_std']['overall'][m]) for m in ('ccc','mae','rmse')])
    sections.append(table(['种子','CCC','MAE','RMSE'],rows))
    sections.append('<h2>结论与限制</h2><ul><li>B01 在这批数据的原始 V/A 指标上优于 V2；六个单元的 CCC 差值区间均大于零。这不能证明目标场景或未知歌曲上的同等收益。</li><li>相对 mixed，PMEmo 唤醒度 CCC 下降，部分单元 MAE 也上升。B01 已未通过内部非劣门槛，本比较不撤销该判定，不替换应用。</li><li>mixed 曾使用本验证集选模；V2 的训练歌曲来源记录不完整，不能声称对 V2 无训练重叠。本验证集已有历史诊断暴露，18 首示例也是依据旧 mixed 诊断固定选择。</li><li>动态变化仍不足。32.7% 是 B01 三种子、六单元平均的“优于自身预测均值常量”比例，不是全体歌曲的加权比例。</li><li>原始预测、去均值诊断分别展示；未应用校准、显示平滑或滞后修正。滞后候选不代表已人工验证。</li><li>人类试听、目标场景多人标注及 Fabio 九首的原始音频核对仍待完成。622 首最终测试未读取，旧应用默认行为未变。</li></ul>')
    compute=report['compute']
    sections.append('<h2>运行与计时范围</h2><p>实际提取 '+str(report['total_v2_frames'])+' 个 V2 特征帧；音频共 '+number(report['total_audio_seconds']/3600,2)+' 小时。V2 在 CPU 执行，mixed/B01 序列头在 GPU 执行并复用既有特征。</p>')
    sections.append('<p>记录的编码累计耗时 '+number(compute['completed_song_encoder_seconds'],1)+' 秒包含系统挂起；恢复后调用耗时 '+number(compute['elapsed_this_invocation_seconds'],1)+' 秒仅覆盖最后一次调用，且不包含失败尝试。不能据此当作纯计算吞吐或公平端到端速度比较。最后一次调用的 PyTorch GPU 峰值 allocated/reserved 为 '+number(compute['peak_torch_gpu_allocated_bytes']/2**20,1)+' / '+number(compute['peak_torch_gpu_reserved_bytes']/2**20,1)+' MiB，不含既有特征提取成本，也不代表整个系统显存占用。</p>')
    sections.append('<p>公共有效标签：V '+str(report['shared_label_values'][0])+'，A '+str(report['shared_label_values'][1])+'；两维各排除 V2 不支持的 112 个末端标签点，所有系统使用相同独立维度掩码。</p>')
    sections.append('<h2>18 首固定示例</h2><p>全程保留原始音频时间轴；无共同标注支持的区间留空，不跨缺失区间连线。紫色粗线为固定集成，淡线为全部三个种子。残差图只用于区分均值偏差与曲线变化。</p>')
    for selection, receipt in zip(protocol['representatives'],receipts):
        slug=selection['song_id'].replace(':','-')
        sections.append('<article><h3>'+esc(selection['song_id']+' · '+selection['reason'])+'</h3><p><a href="curves/'+slug+'.svg">矢量图</a> · <a href="curves/'+slug+'.json">原始图表数据与滞后标记</a></p><img loading="lazy" src="curves/'+slug+'.png" alt="'+esc(selection['song_id'])+' raw V/A, residual and lag comparison"></article>')
    sections.append('<footer><p>源报告 SHA256：'+sha256(SOURCE/'report.json')+'</p><p>每首图的输入、输出 SHA256 和重现命令见本目录 launch.json / report.json / verification.json。运行中断记录：'+esc('; '.join(launch['runtime_notes']))+'</p></footer>')
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>584 首连续情绪跨版本比较</title><style>body{max-width:1320px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#233345;background:#fafbfc}h1{font-size:32px}h2{margin-top:44px}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{padding:8px 10px;border:1px solid #dce3e8;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef3f8}article{background:white;padding:20px;margin:28px 0;border:1px solid #dce3e8}img{width:100%;height:auto}footer{font-size:12px;overflow-wrap:anywhere;color:#5b6a78}a{color:#246b9a}</style><body>'+''.join(sections)+'</body></html>'


def inactive(path):
    if path.exists() and process_alive(read_json(path)['process_identity']) is not False:
        raise ValueError('Prior process identity still active or unknown')


def run(launch, resume):
    if (DIRECTORY/'report.json').exists():
        raise ValueError('Preserve completed render; use --verify')
    if (DIRECTORY/'process.json').exists() and not resume:
        raise ValueError('Existing attempt requires --resume after diagnosis')
    inactive(DIRECTORY/'process.json')
    atomic_json(DIRECTORY/'process.json',dict(process_identity=process_identity(),started_at=now(),resumed=resume))
    record_event(dict(event='F_comparison_renderer_started',job_id=JOB),run=dict(run_id=JOB,stage='F',status='running'))
    protocol,report,_=checked_inputs(); plt,versions=setup_plotting(); _,outputs=stores()
    metrics={r['song_id']:r['scores'] for r in report['songs']}; receipts=[]
    for selection in protocol['representatives']:
        index,row=next((i,r) for i,r in enumerate(protocol['validation_songs']) if r['song_id']==selection['song_id'])
        key=key_for(index,row); source_receipt,saved=outputs.read(key,row['song_id'])
        slug=row['song_id'].replace(':','-'); receipt_path=DIRECTORY/'curves'/(slug+'.receipt.json')
        if receipt_path.exists():
            receipt=read_json(receipt_path); verify_hashes(receipt['files'])
            if receipt['source_prediction_sha256']!=source_receipt['sha256'] or receipt['launch_sha256']!=sha256(DIRECTORY/'launch.json'):
                raise ValueError('Rendered figure input binding changed')
        else:
            paths=[DIRECTORY/'curves'/(slug+suffix) for suffix in ('.json','.png','.svg')]
            if any(path.exists() or path.with_suffix(path.suffix+'.tmp').exists() for path in paths):
                raise ValueError('Preserve uncommitted render for inspection: '+slug)
            payload=curve_payload(selection,row,saved,metrics[row['song_id']]); atomic_json(paths[0],payload)
            fig=figure(plt,payload)
            for path,fmt in zip(paths[1:],('png','svg')):
                temporary=path.with_suffix(path.suffix+'.tmp'); fig.savefig(temporary,format=fmt,dpi=140)
                temporary.replace(path)
            plt.close(fig)
            receipt=dict(song_id=row['song_id'],completed_at=now(),source_prediction_sha256=source_receipt['sha256'],
                         launch_sha256=sha256(DIRECTORY/'launch.json'),files={rel(p):sha256(p) for p in paths})
            atomic_json(receipt_path,receipt)
        receipts.append(receipt); state('rendered_song',song_id=row['song_id'],completed=len(receipts),total=18)
    index=DIRECTORY/'index.html'
    if index.exists(): raise ValueError('Preserve uncommitted index for inspection')
    index.write_text(document(protocol,report,launch,receipts),encoding='utf-8')
    result=dict(job_id=JOB,status='rendered_pending_verification',completed_at=now(),versions=versions,
                source_report_sha256=sha256(SOURCE/'report.json'),launch_sha256=sha256(DIRECTORY/'launch.json'),
                representatives=protocol['representatives'],receipts=receipts,index_sha256=sha256(index),
                visual_review='pending',final_test_songs_used=0,model_changes=False,native_gap_check=check_segments())
    atomic_json(DIRECTORY/'report.json',result)
    record_event(dict(event='F_comparison_renderer_finished',job_id=JOB,report=rel(DIRECTORY/'report.json')),
                 run=dict(run_id=JOB,stage='F',status='pending_verification',report=rel(DIRECTORY/'report.json')))
    state('rendered_pending_verification',completed=18)


def verify(launch):
    if (DIRECTORY/'verification.json').exists(): raise ValueError('Preserve passed renderer verification')
    inactive(DIRECTORY/'process.json'); inactive(DIRECTORY/'verification-process.json')
    atomic_json(DIRECTORY/'verification-process.json',dict(process_identity=process_identity(),started_at=now()))
    protocol,source,_=checked_inputs(); report=read_json(DIRECTORY/'report.json'); _,outputs=stores()
    if report['source_report_sha256']!=sha256(SOURCE/'report.json') or report['launch_sha256']!=sha256(DIRECTORY/'launch.json'):
        raise ValueError('Renderer provenance changed')
    if report['representatives']!=protocol['representatives'] or len(report['receipts'])!=18:
        raise ValueError('Representative selection changed')
    setup_plotting(); from PIL import Image
    scores={r['song_id']:r['scores'] for r in source['songs']}
    for selection,receipt in zip(protocol['representatives'],report['receipts']):
        assert selection['song_id']==receipt['song_id']; verify_hashes(receipt['files'])
        index,row=next((i,r) for i,r in enumerate(protocol['validation_songs']) if r['song_id']==selection['song_id'])
        source_receipt,saved=outputs.read(key_for(index,row),row['song_id'])
        assert source_receipt['sha256']==receipt['source_prediction_sha256']
        stem=DIRECTORY/'curves'/row['song_id'].replace(':','-')
        payload=read_json(stem.with_suffix('.json'))
        assert payload==curve_payload(selection,row,saved,scores[row['song_id']])
        for a,axis in enumerate(AXES):
            data=payload['axes'][axis]; t=np.asarray(data['times']); valid=saved['common_mask'][:,a]
            np.testing.assert_array_equal(t,saved['times'][valid]); np.testing.assert_array_equal(data['truth'],saved['y'][valid,a])
            for s,name in enumerate(SYSTEMS): np.testing.assert_array_equal(data['predictions'][name],saved['predictions'][s,valid,a])
            covered=[i for part in data['segments'] for i in part]; assert covered==list(range(len(t)))
            for part in data['segments']: assert np.all(np.diff(t[part]) <= payload['native_seconds']+1e-7)
        with Image.open(stem.with_suffix('.png')) as im:
            assert im.size==(2240,1680); im.verify()
        assert ET.parse(stem.with_suffix('.svg')).getroot().tag.endswith('svg')
    assert sha256(DIRECTORY/'index.html')==report['index_sha256']
    assert (DIRECTORY/'index.html').read_text(encoding='utf-8')==document(protocol,source,launch,report['receipts'])
    result=dict(job_id=JOB,status='passed',verified_at=now(),verified_songs=18,raw_arrays_exact=True,
                independent_va_masks_preserved=True,missing_intervals_not_bridged=True,images_decoded=18,svg_parsed=18,
                source_report_sha256=sha256(SOURCE/'report.json'),report_sha256=sha256(DIRECTORY/'report.json'),
                visual_layout_review='pending',final_test_songs_used=0)
    atomic_json(DIRECTORY/'verification.json',result)
    record_event(dict(event='F_comparison_renderer_verified',**result),run=dict(run_id=JOB,stage='F',status='completed_verified',verification=rel(DIRECTORY/'verification.json')))
    state('completed_verified',verified_songs=18)


def status():
    path=DIRECTORY/('verification-process.json' if (DIRECTORY/'verification-process.json').exists() else 'process.json')
    identity=read_json(path)['process_identity'] if path.exists() else None
    return dict(job_id=JOB,checked_at=now(),os_lease_active=is_running(JOB),process_identity=identity,
                process_alive=process_alive(identity) if identity else None,
                progress=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None,
                receipts=len(list((DIRECTORY/'curves').glob('*.receipt.json'))))


def main():
    parser=argparse.ArgumentParser(description=__doc__); modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true'); modes.add_argument('--check',action='store_true'); modes.add_argument('--status',action='store_true'); modes.add_argument('--launch',type=Path)
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--verify',action='store_true'); args=parser.parse_args()
    if args.status: print(json.dumps(status())); return
    with run_lease(JOB):
        (DIRECTORY/'curves').mkdir(parents=True,exist_ok=True)
        if args.check: check(); return
        if args.prepare: prepare(); return
        if args.launch.resolve()!=DIRECTORY/'launch.json': raise ValueError('Wrong renderer launch file')
        launch=read_json(args.launch)
        if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(launch['python']).resolve(): raise ValueError('Wrong registered Python or cwd')
        if any(os.environ.get(k)!=v for k,v in launch['environment'].items()): raise ValueError('Frozen renderer environment changed')
        verify_hashes(launch['pinned_hashes']); verify_hashes(read_json(SOURCE/'launch.json')['pinned_hashes'])
        with (DIRECTORY/'console.log').open('a',encoding='utf-8',buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
                try:
                    if args.verify: verify(launch)
                    else: run(launch,args.resume)
                except Exception as exc:
                    failure=dict(at=now(),error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
                    atomic_json(DIRECTORY/('failure-'+str(process_identity()['pid'])+'.json'),failure)
                    state('failed',error=failure['error'])
                    record_event(dict(event='F_comparison_renderer_failed',job_id=JOB,**failure),run=dict(run_id=JOB,stage='F',status='failed'))
                    raise


if __name__=='__main__': main()
