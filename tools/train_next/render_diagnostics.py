"""Render saved validation diagnostics as an offline HTML report, PNGs and PDF."""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
AXES = ('valence', 'arousal')
NAMES = {'valence': '效价 V', 'arousal': '唤醒 A'}
DOMAINS = {'deam': 'DEAM', 'memo2496': 'Memo2496', 'pmemo': 'PMEmo'}
TRUE, PREDICTED, INK, MUTED = '#16877d', '#d9653b', '#233345', '#75818c'


def number(value, signed=False):
    return '—' if value is None else f'{value:+.3f}' if signed else f'{value:.3f}'


def data_uri(path):
    return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def setup_plotting():
    cache = ROOT / 'data/training/next_va/diagnostics/.matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager, pyplot as plt
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams['font.family'] = [font, 'DejaVu Sans']
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                         'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff',
                         'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.edgecolor': '#d4dce1', 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    return plt


def segments(times):
    if len(times) < 2:
        return [np.arange(len(times))]
    interval = np.median(np.diff(times))
    return np.split(np.arange(len(times)), np.flatnonzero(np.diff(times) > interval * 1.5) + 1)


def song_figure(plt, song, curve, selection, zoom=False):
    times = np.asarray(curve['times'])
    predictions = np.asarray(curve['prediction'])
    truth = np.array([[np.nan if value is None else value for value in row] for row in curve['truth']])
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [2.5, 1]})
    title = str(song.get('title') or '未提供标题')
    if len(title) > 65:
        title = title[:62] + '…'
    fig.suptitle(f"{DOMAINS[song['source_dataset']]} / {song['song_id']} · {selection['reason']}\n{title}",
                 x=.06, y=.98, ha='left', fontsize=16, fontweight='bold')
    for index, axis in enumerate(AXES):
        panel, lag_panel = axes[index]
        metrics = song[axis]
        valid = np.isfinite(truth[:, index])
        observed_times = times[valid]
        panel.plot(times, predictions[:, index], color=PREDICTED, lw=1.5, label='预测曲线（原始、未平滑）')
        for k, part in enumerate(segments(observed_times)):
            panel.plot(observed_times[part], truth[valid, index][part], color=TRUE, lw=1.9,
                       marker='.', ms=2, label='真实标注' if k == 0 else None)
        panel.axhline(metrics['truth_mean'], color=TRUE, lw=1, ls='--', alpha=.75, label='真实均值')
        panel.axhline(metrics['prediction_mean'], color=PREDICTED, lw=1, ls='--', alpha=.75, label='预测均值')
        # Shade only genuinely unobserved regions, not Memo2496's half-second gaps.
        parts = segments(observed_times)
        covered = [(observed_times[p[0]], observed_times[p[-1]]) for p in parts if len(p)]
        cursor = 0.
        for start, end in covered:
            if start > cursor:
                panel.axvspan(cursor, start, color='#e7ebee', alpha=.65, lw=0)
            cursor = end
        if cursor < song['duration_seconds']:
            panel.axvspan(cursor, song['duration_seconds'], color='#e7ebee', alpha=.65, lw=0)
        if zoom:
            values = np.r_[predictions[:, index], truth[valid, index]]
            lo, hi = values.min(), values.max()
            margin = max(.025, (hi - lo) * .12)
            panel.set_ylim(max(-1.05, lo - margin), min(1.05, hi + margin))
        else:
            panel.set_ylim(-1.05, 1.05)
        panel.set_xlim(0, song['duration_seconds'])
        panel.grid(alpha=.16)
        panel.set_xlabel('音频时间 / 秒')
        panel.set_ylabel('局部纵轴' if zoom else '统一范围 [−1, 1]')
        panel.set_title(f"{NAMES[axis]}    CCC {number(metrics['ccc'])}    MAE {number(metrics['mae'])}\n"
                        f"均值 真 {number(metrics['truth_mean'])} / 预 {number(metrics['prediction_mean'])}   "
                        f"偏差 {number(metrics['mean_bias'], True)}   "
                        f"振幅比 σ预/σ真 {number(metrics['std_ratio'])}", loc='left', fontsize=10.5, pad=12)
        if index == 0:
            panel.legend(loc='lower right', ncol=2, fontsize=8, framealpha=.9)
        lag = metrics['lag']
        search = lag['search_seconds']
        if search:
            correlations = [np.nan if v is None else v for v in lag['correlations']]
            lag_panel.plot(search, correlations, color='#586a96', marker='o', ms=3, lw=1.4)
            if lag['best_seconds'] is not None:
                lag_panel.axvline(lag['best_seconds'], color=PREDICTED, ls='--', lw=1)
                lag_panel.scatter([lag['best_seconds']], [lag['best_correlation']], color=PREDICTED, s=38, zorder=5)
        lag_panel.axvline(0, color=MUTED, alpha=.6, lw=.8)
        lag_panel.axhline(0, color=MUTED, alpha=.35, lw=.8)
        lag_panel.set_ylim(-1.05, 1.05)
        lag_panel.grid(alpha=.16)
        lag_panel.set_xlabel('候选移位 / 秒（正值：预测较晚）')
        lag_panel.set_ylabel('Pearson 相关系数')
        lag_text = '不可识别' if lag['best_seconds'] is None else f"{lag['best_seconds']:+.1f} 秒"
        lag_panel.set_title(f"候选时延 {lag_text}\n{lag['status']}", loc='left', fontsize=10.5, pad=12)
        lag_panel.text(.03, .04, f"相同 {lag['points']} 个标注点\n相关改善 {number(lag['improvement'], True)}",
                       transform=lag_panel.transAxes, fontsize=8, color=MUTED,
                       bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    fig.text(.06, .027, '灰色：无真实标注。均值、振幅、误差仅使用有标注的位置；实线未做移位或平滑。\n'
             '时延为诊断性相关峰，不等于处理延迟；低可信峰不能用于统一平移标签。', fontsize=9, color=MUTED)
    fig.subplots_adjust(left=.07, right=.97, top=.81, bottom=.13, hspace=.7, wspace=.30)
    return fig


def overview_figure(plt, report):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle('全部验证歌曲：均值偏差与变化幅度', x=.06, ha='left', fontsize=18, fontweight='bold', y=.98)
    fig.text(.06, .915, '每个点代表一首歌 · 效价 / 唤醒独立统计 · 18 首例图以外的歌曲也纳入分布', color=MUTED, fontsize=11)
    rng = np.random.default_rng(20261003)
    all_ratios = [song[axis]['std_ratio'] for song in report['songs'] for axis in AXES
                  if song[axis]['std_ratio'] is not None and song[axis]['std_ratio'] > 0]
    ratio_limits = (min(all_ratios) * .8, max(all_ratios) * 1.5)
    for column, (domain, summary) in enumerate(report['domain_summaries'].items()):
        rows = [r for r in report['songs'] if r['source_dataset'] == domain]
        for i, axis in enumerate(AXES):
            color = TRUE if i == 0 else PREDICTED
            bias = [r[axis]['mean_bias'] for r in rows]
            jitter = rng.uniform(-.14, .14, len(bias))
            axes[0, column].scatter(i + jitter, bias, s=9, color=color, alpha=.4, edgecolors='none')
            axes[0, column].plot([i - .19, i + .19], [np.median(bias)] * 2, color=INK, lw=2)
            ratios = [r[axis]['std_ratio'] for r in rows if r[axis]['std_ratio'] is not None and r[axis]['std_ratio'] > 0]
            axes[1, column].scatter(i + rng.uniform(-.14, .14, len(ratios)), ratios, s=9, color=color, alpha=.4, edgecolors='none')
            axes[1, column].plot([i - .19, i + .19], [np.median(ratios)] * 2, color=INK, lw=2)
            axes[1, column].text(i, .96, f"中位数 {np.median(ratios):.2f}×", transform=axes[1, column].get_xaxis_transform(),
                                  ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
        axes[0, column].set_title(f"{DOMAINS[domain]} · {summary['songs']} 首", fontsize=13, loc='left', pad=12)
        axes[0, column].set_ylim(-1.05, 1.05)
        axes[0, column].axhline(0, color=MUTED, ls='--', lw=1)
        axes[1, column].set_yscale('log')
        axes[1, column].set_ylim(*ratio_limits)
        axes[1, column].axhline(1, color=MUTED, ls='--', lw=1)
        for row in range(2):
            axes[row, column].set_xlim(-.55, 1.55)
            axes[row, column].set_xticks([0, 1], ['效价 V', '唤醒 A'])
            axes[row, column].grid(axis='y', alpha=.17)
    axes[0, 0].set_ylabel('均值偏差：预测 − 真实')
    axes[1, 0].set_ylabel('振幅比 σ预测 / σ真实（对数轴）')
    fig.text(.06, .035, '均值偏差越接近 0 越好；振幅比大于 1 表示预测波动更大，小于 1 表示更平。黑线为中位数。\n'
             '所有指标使用原始曲线与有效标注；接近常量、无法稳定计算振幅比的曲线不纳入该比值。', fontsize=9, color=MUTED)
    fig.subplots_adjust(top=.83, bottom=.14, left=.08, right=.97, wspace=.25, hspace=.34)
    return fig


def table_for_song(song):
    cells = []
    for axis in AXES:
        m = song[axis]
        lag = m['lag']
        delay = '不可识别' if lag['best_seconds'] is None else f"{lag['best_seconds']:+.1f} 秒"
        cells.append('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in [
            NAMES[axis], number(m['truth_mean']), number(m['prediction_mean']), number(m['mean_bias'], True),
            number(m['truth_span90']), number(m['prediction_span90']), number(m['std_ratio']),
            number(m['ccc']), number(m['mae']), delay]) + '</tr>')
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{v}</th>' for v in [
        '维度', '真实均值', '预测均值', '均值偏差', '真实 P90 振幅', '预测 P90 振幅', 'σ 振幅比', 'CCC', 'MAE', '候选时延']) + '</tr></thead><tbody>' + ''.join(cells) + '</tbody></table></div>'


def write_csvs(output, report):
    selected = {r['song_id']: r['reason'] for r in report['representatives']}
    fields = ['source_dataset', 'song_id', 'title', 'artist', 'selection', 'axis', 'values', 'truth_mean',
              'prediction_mean', 'mean_bias', 'truth_std', 'prediction_std', 'std_ratio', 'truth_span90',
              'prediction_span90', 'mae', 'rmse', 'ccc', 'constant_prediction_mae', 'centered_rmse',
              'lag_seconds', 'lag_correlation', 'lag_improvement', 'lag_status']
    with (output / 'all-validation-metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for song in report['songs']:
            for axis in AXES:
                m = song[axis]
                row = {key: song.get(key, '') for key in fields[:4]}
                row.update({key: m[key] for key in fields if key in m})
                row.update(axis=axis, selection=selected.get(song['song_id'], ''),
                           lag_seconds=m['lag']['best_seconds'], lag_correlation=m['lag']['best_correlation'],
                           lag_improvement=m['lag']['improvement'], lag_status=m['lag']['status'])
                writer.writerow(row)
    with (output / 'representative-curves.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['song_id', 'time_seconds', 'true_valence', 'true_arousal', 'predicted_valence', 'predicted_arousal'])
        for key, curve in report['curves'].items():
            for t, truth, prediction in zip(curve['times'], curve['truth'], curve['prediction']):
                writer.writerow([key, t, *truth, *prediction])


def render(args):
    output = args.directory.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Report output must stay inside Vm')
    report = json.loads((output / 'diagnostics.json').read_text(encoding='utf-8'))
    plt = setup_plotting()
    from matplotlib.backends.backend_pdf import PdfPages
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    by_id = {s['song_id']: s for s in report['songs']}
    cards = []
    with PdfPages(output / 'representative-curves.pdf') as pdf:
        overview = overview_figure(plt, report)
        overview.savefig(output / 'overview.png', dpi=150)
        pdf.savefig(overview)
        plt.close(overview)
        for i, selection in enumerate(report['representatives'], 1):
            key = selection['song_id']
            song = by_id[key]
            paths = []
            for zoom in (False, True):
                fig = song_figure(plt, song, report['curves'][key], selection, zoom)
                path = figures / (key.replace(':', '-') + ('-detail.png' if zoom else '.png'))
                fig.savefig(path, dpi=135)
                if not zoom:
                    pdf.savefig(fig)
                plt.close(fig)
                paths.append(path)
            title = html.escape(str(song.get('title') or '未提供标题'))
            artist = html.escape(str(song.get('artist') or '未提供艺人'))
            warnings = ' · '.join(f"{NAMES[a]}：{song[a]['lag']['status']}" for a in AXES)
            cards.append(f'''<article class="song" data-domain="{song['source_dataset']}" data-category="{selection['category']}" id="song-{key.replace(':', '-')}">
              <div class="song-head"><div><span class="tag">{DOMAINS[song['source_dataset']]}</span><span class="tag light">{selection['reason']}</span>
              <h3>{title}</h3><p>{html.escape(key)} · {artist} · {song['duration_seconds']:.1f} 秒</p></div>
              <div class="ranks">真实变化分位 P{selection['variation_percentile'] * 100:.0f}<br>双维 CCC 分位 P{selection['quality_percentile'] * 100:.0f}</div></div>
              <a class="figure-link" href="{paths[0].relative_to(output).as_posix()}" target="_blank" title="打开原尺寸图片">
              <img class="full-plot" src="{data_uri(paths[0])}" alt="{key} 效价、唤醒真实预测曲线及相关移位图" loading="lazy">
              <img class="detail-plot" src="{data_uri(paths[1])}" alt="{key} 局部纵轴曲线" loading="lazy" hidden></a>
              {table_for_song(song)}<p class="lag-note">{html.escape(warnings)}</p></article>''')
            print(json.dumps(dict(rendered=i, total=len(report['representatives']), song_id=key)), flush=True)
    summaries = []
    for domain, group in report['domain_summaries'].items():
        for axis in AXES:
            m = group[axis]
            fraction = m['dynamics_beats_constant'] / group['songs']
            summaries.append(f"<tr><td>{DOMAINS[domain]}</td><td>{NAMES[axis]}</td><td>{group['songs']}</td>"
                             f"<td>{m['ccc']:.3f}</td><td>{m['mae']:.3f}</td><td>{m['mean_absolute_bias']:.3f}</td>"
                             f"<td>{m['median_std_ratio']:.2f}×</td><td>{fraction:.0%}（{m['dynamics_beats_constant']}/{group['songs']}）</td></tr>")
    options = ''.join(f'<option value="{key}">{name}</option>' for key, name in DOMAINS.items())
    categories = [('typical', '典型歌曲'), ('steady', '较平稳'), ('dynamic', '变化明显'), ('better', '拟合较好'), ('worse', '拟合较差'), ('long', '较长歌曲')]
    category_options = ''.join(f'<option value="{key}">{name}</option>' for key, name in categories)
    document = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>音乐情绪模型 · 真实与预测诊断</title><style>
    :root{color-scheme:light;--ink:#233345;--muted:#687680;--teal:#16877d;--orange:#d9653b;--line:#e1e6e8}*{box-sizing:border-box}
    body{margin:0;background:#f4f5f1;color:var(--ink);font:15px/1.65 "Microsoft YaHei",system-ui,sans-serif}
    main{max-width:1440px;margin:auto;padding:44px 32px 72px}header{border-bottom:1px solid var(--line);padding-bottom:26px}
    .eyebrow{font-size:12px;letter-spacing:.14em;color:var(--teal);font-weight:700}h1{font-size:36px;line-height:1.25;margin:14px 0}h2{font-size:23px;margin:0 0 12px}h3{font-size:22px;line-height:1.35;margin:12px 0 6px}
    p{margin:8px 0;color:var(--muted)}a{color:var(--teal)}.lead{font-size:17px;max-width:1000px}.meta{font-size:12px;word-break:break-all}
    .findings{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}.finding{background:#fff;padding:22px;border:1px solid var(--line);border-radius:13px}.finding strong{display:block;font-size:24px;margin:6px 0}.finding p{font-size:13px}
    section{margin-top:30px}.overview,.methods{background:#fff;border:1px solid var(--line);border-radius:14px;padding:24px}.overview img{width:100%;display:block}
    .table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th{text-align:left;background:#f4f7f6;font-weight:600}td,th{padding:11px 13px;border-bottom:1px solid var(--line)}td{font-variant-numeric:tabular-nums}
    .controls{position:sticky;top:0;z-index:3;background:#f4f5f1f5;backdrop-filter:blur(10px);border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:18px 0;margin-top:24px}
    select,button{font:inherit;background:#fff;border:1px solid #cbd6d8;border-radius:8px;padding:9px 14px;color:var(--ink)}button{cursor:pointer}button.active{background:var(--ink);color:white}.count{margin-left:auto;font-size:13px;color:var(--muted)}
    .song{background:white;border:1px solid var(--line);border-radius:14px;margin-top:22px;overflow:hidden;scroll-margin-top:90px}.song-head{display:flex;justify-content:space-between;gap:20px;padding:24px 28px 0}.tag{display:inline-block;background:#e5f1ef;color:#15786f;font-size:12px;padding:3px 9px;border-radius:5px;margin-right:8px}.tag.light{background:#f4ede7;color:#9b593b}.ranks{text-align:right;white-space:nowrap;font-size:12px;color:var(--muted);padding-top:7px}
    .figure-link{display:block}.song img{width:100%;display:block}.song img[hidden]{display:none}.song .table-wrap{margin:0 24px}.lag-note{margin:16px 28px 24px;font-size:12px;color:#856341}article[hidden]{display:none}
    .downloads{display:flex;gap:20px;flex-wrap:wrap;margin-top:16px}.methods li{margin-bottom:8px}.legend{display:flex;gap:22px;font-size:13px}.legend b{display:inline-block;width:22px;height:3px;vertical-align:middle;margin-right:7px}.notes{font-size:13px}footer{margin-top:32px;font-size:12px;color:var(--muted)}
    @media(max-width:760px){main{padding:24px 14px}.findings{grid-template-columns:1fr}h1{font-size:28px}.song-head{padding:18px 16px 0}.ranks{display:none}.overview,.methods{padding:16px}.count{margin-left:0}select{max-width:160px}}
    </style></head><body><main>
    <header><div class="eyebrow">MUSIC EMOTION / VALIDATION DIAGNOSTICS</div><h1>真实情绪与模型预测，逐曲对照</h1>
    <p class="lead">DEAM、Memo2496、PMEmo 每库 6 首代表歌曲。展示原始曲线、均值、变化幅度和候选时延，并提供全部 584 首验证歌曲的统计。</p>
    <p class="meta">模型：mixed-20261003-recovery / best.pth · 第 15 轮 · 验证集诊断 · 未使用测试集</p>
    <div class="downloads"><a href="representative-curves.pdf">下载图册 PDF（19 页）</a><a href="all-validation-metrics.csv">全部歌曲指标 CSV</a><a href="representative-curves.csv">18 首逐时刻曲线 CSV</a><a href="diagnostics.json">完整诊断 JSON</a></div></header>
    <div class="findings"><div class="finding"><span class="eyebrow">01 / 变化幅度</span><strong>预测总体波动偏大</strong><p>六组“数据集 × 维度”的 σ 振幅比中位数均大于 1，约 1.21–1.81 倍。当前问题并非整体曲线过平。</p></div>
    <div class="finding"><span class="eyebrow">02 / 歌曲基调</span><strong>逐曲均值仍有偏差</strong><p>逐曲均值绝对偏差约 0.12–0.19，与 MAE 接近。正负偏差会相互抵消，不能只看全库平均偏差。</p></div>
    <div class="finding"><span class="eyebrow">03 / 动态收益</span><strong>约 22%–41% 的歌曲受益</strong><p>按维度与数据集统计，只有这些歌曲的动态曲线 MAE 优于“该曲预测均值”的常量输出。变化的方向与时机需要继续改进。</p></div></div>
    <section class="overview"><h2>全库统计</h2><p>下面使用全部验证歌曲，避免把代表例图当作总体表现。振幅比接近 1 只表示变化大小接近，并不保证曲线方向正确。</p>
    <div class="table-wrap"><table><thead><tr><th>数据集</th><th>维度</th><th>歌曲数</th><th>CCC ↑</th><th>MAE ↓</th><th>均值绝对偏差 ↓</th><th>σ 振幅比中位数</th><th>动态优于常量的歌曲</th></tr></thead><tbody>__SUMMARY__</tbody></table></div>
    <img src="__OVERVIEW__" alt="全部验证歌曲的均值偏差和振幅比分布"></section>
    <section><h2>18 首代表歌曲</h2><div class="legend"><span><b style="background:#16877d"></b>真实标注</span><span><b style="background:#d9653b"></b>模型预测</span><span>虚线：各自均值</span><span>灰色：无真实标注</span></div>
    <div class="controls"><label>数据集 <select id="domain"><option value="all">全部数据集</option>__DOMAINS__</select></label><label>类型 <select id="category"><option value="all">全部类型</option>__CATEGORIES__</select></label>
    <button id="scale" aria-pressed="false">局部纵轴：关闭</button><span class="count" id="count">显示 18 首</span></div>__CARDS__</section>
    <section class="methods"><h2>如何读图</h2><ol>
    <li><b>均值：</b>只在真实标注有效的位置计算两条曲线的均值。偏差 = 预测 − 真实。灰色区域虽然有预测，但不参与误差统计。</li>
    <li><b>振幅：</b>P90 振幅 = 第 95 百分位 − 第 5 百分位，降低单点尖峰影响；σ 振幅比 = 预测标准差 / 真实标准差。低变化曲线的比值应结合绝对幅度阅读。</li>
    <li><b>时延：</b>计算真实 y(t) 与预测 p(t + 移位) 的相关。正值表示预测相对标注较晚，负值表示较早。搜索最多 ±8 秒，短片段缩小到有效标注时长的四分之一；所有移位使用相同的中间标注点。相关峰不是处理延迟，也不是经显著性检验的结论。</li>
    <li><b>可靠性：</b>真实 σ &lt; 0.02、预测 σ &lt; 0.005、峰值相关 &lt; 0.3、峰值在边界、移位改善 &lt; 0.05 等情况会提示不可靠或无明显证据。以上为公开的诊断启发式，不是置信区间。</li>
    <li><b>抽样：</b>按双维真实波动与 CCC 的库内分位，选择典型、较平稳（约 P15）、变化明显（约 P85）、拟合较好（约 P85）、拟合较差（约 P15），再取未选歌曲中最长的一首。例图按此规则选出，不是随机样本。</li>
    <li><b>时间轴：</b>使用发布音频自身的时间。PMEmo 是副歌片段，DEAM 部分也是片段；没有把片段冒充完整原曲。Memo2496 保留原生 1 Hz 标注，没有插值制造标签；不同标注段之间不连线。</li>
    <li><b>显示：</b>默认所有曲线使用 [−1,1] 统一纵轴；“局部纵轴”放大实际数值范围，适合观察细小变化。预测不平滑、不移位，CCC 与 MAE 均是原始对齐分数。</li>
    </ol></section><footer>生成时间：__TIME__<br>模型 SHA256：__HASH__<br>已复算全部检查点指标；最大数值差 __DIFF__。原模型保持不变。本报告离线可用，不加载外部脚本或字体。</footer>
    </main><script>
    const domain=document.querySelector('#domain'),category=document.querySelector('#category'),scale=document.querySelector('#scale');
    function filter(){let n=0;document.querySelectorAll('.song').forEach(card=>{card.hidden=!((domain.value==='all'||card.dataset.domain===domain.value)&&(category.value==='all'||card.dataset.category===category.value));if(!card.hidden)n++;});document.querySelector('#count').textContent=`显示 ${n} 首`;}
    domain.addEventListener('change',filter);category.addEventListener('change',filter);
    scale.addEventListener('click',()=>{const active=scale.getAttribute('aria-pressed')!=='true';scale.setAttribute('aria-pressed',String(active));scale.classList.toggle('active',active);scale.textContent=`局部纵轴：${active?'开启':'关闭'}`;document.querySelectorAll('.full-plot').forEach(el=>el.hidden=active);document.querySelectorAll('.detail-plot').forEach(el=>el.hidden=!active);document.querySelectorAll('.figure-link').forEach(a=>{a.href=active?a.getAttribute('href').replace(/(?:-detail)?\.png$/,'-detail.png'):a.getAttribute('href').replace('-detail.png','.png');});});
    </script></body></html>'''
    replacements = {'__SUMMARY__': ''.join(summaries), '__OVERVIEW__': data_uri(output / 'overview.png'),
                    '__DOMAINS__': options, '__CATEGORIES__': category_options, '__CARDS__': ''.join(cards),
                    '__TIME__': html.escape(report['generated_at']), '__HASH__': report['checkpoint_sha256'],
                    '__DIFF__': f"{report['max_metric_difference']:.3g}"}
    for key, value in replacements.items():
        document = document.replace(key, value)
    (output / 'index.html').write_text(document, encoding='utf-8')
    write_csvs(output, report)
    print(json.dumps(dict(status='rendered', report=str(output / 'index.html'), pdf=str(output / 'representative-curves.pdf'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    render(parser.parse_args())
