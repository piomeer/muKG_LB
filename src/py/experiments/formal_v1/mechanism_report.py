"""Verify the complete mechanism matrix and report exploratory paired effects."""
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from . import diagnostics as d
from .mechanism_sampling import GROUPS

SEEDS=(401,402,403)


def matrix_statistics(matrix):
    from .protocol import paired_summary
    missing=[f'V-{group}-{seed}' for seed in SEEDS for group in GROUPS if group not in matrix.get(str(seed),{})]
    if missing:
        return {'status':'incomplete','missing':missing,'candidate':None}
    if any(not math.isfinite(value) or value<=0 for row in matrix.values() for value in row.values()):
        raise ValueError('MRR matrix requires positive finite values')
    comparisons={}
    for left,right in (('M10','M00'),('M01','M00'),('M11','M01'),('M11','M10'),('M11','M00')):
        comparisons[f'{left}/{right}']=paired_summary([matrix[str(seed)][left]/matrix[str(seed)][right] for seed in SEEDS])
    effects={name:[] for name in ('direction','uniqueness','interaction')}
    for seed in SEEDS:
        row=matrix[str(seed)]
        log00,log10,log01,log11=(math.log(row[group]) for group in GROUPS)
        effects['direction'].append(math.exp(.5*((log10-log00)+(log11-log01))))
        effects['uniqueness'].append(math.exp(.5*((log01-log00)+(log11-log10))))
        effects['interaction'].append(math.exp((log11-log01)-(log10-log00)))
    averages={group:mean(matrix[str(seed)][group] for seed in SEEDS) for group in GROUPS}
    candidate=max(GROUPS,key=averages.get)
    return {'status':'complete','comparisons':comparisons,
            'effects':{name:paired_summary(values) for name,values in effects.items()},
            'group_means':averages,'candidate':candidate,
            'new_candidate_found':candidate!='M00','selection':'mean of seed-wise best overall validation MRR; ties M00/M10/M01/M11'}


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def rank_metrics(metrics, expected=17535):
    for side in ('head','tail'):
        ranks=metrics['ranks'][side]
        if len(ranks)!=expected or any(not 1<=rank<=14541 or rank*2!=int(rank*2) for rank in ranks):
            raise ValueError('Invalid or incomplete averaged validation ranks')
    for side in ('head','tail','overall'):
        ranks=metrics['ranks']['head']+metrics['ranks']['tail'] if side=='overall' else metrics['ranks'][side]
        array=np.asarray(ranks,dtype=np.float64)
        actual={'mrr':float(np.mean(1/array)),**{f'hits@{k}':float(np.mean(array<=k)) for k in (1,3,10)}}
        if any(not math.isclose(value,metrics[side][key],abs_tol=1e-12,rel_tol=0) for key,value in actual.items()):
            raise ValueError('Validation metrics differ from raw ranks')


def verify_job(root,task,outcome,record):
    import torch
    path=(root/outcome['path']).resolve()
    if path.parent!=root/'jobs'/task['id'] or not path.name.startswith('attempt-'):
        raise ValueError('Mechanism result path outside authorized task')
    if d.sha(path/'artifacts.json')!=outcome['artifacts_sha256']:
        raise ValueError('Changed result inventory anchor')
    d.verify_seal(path)
    for name in ('request.json','stdout.log','stderr.log'):
        if d.sha(path.with_name(path.name+'.'+name))!=outcome['sidecars'][name]:
            raise ValueError('Changed task request/log')
    state=d.read(path/'status.json')
    manifest=d.read(path/'manifest.json')
    if state['status']!='complete' or any(outcome.get(key)!=value for key,value in state.items()):
        raise ValueError('Outcome differs from complete state')
    if manifest['task']!=task or any(manifest.get(key)!=record[key] for key in ('config','environment','provenance','freeze_sha256')):
        raise ValueError('Worker manifest differs from protocol')
    if (path/'test_metrics.json').exists():
        raise ValueError('Mechanism task contains test output')
    if task['kind']!='validation':
        expected=15 if task['kind']=='observe_a' else 180
        rows=[json.loads(line) for line in (path/'observations.jsonl').read_text().splitlines()]
        if state['observation_calls']!=expected or len(rows)!=expected:
            raise ValueError('Incomplete observation cells')
        return {'state':state,'path':outcome['path'],'observations':rows}
    epochs=read_csv(path/'epochs.csv')
    validations=read_csv(path/'validation.csv')
    if state['epochs_completed']!=1000 or len(epochs)!=1000 or state['validation_count']!=100:
        raise ValueError('Incomplete validation training')
    if [int(row['epoch']) for row in epochs]!=list(range(1,1001)) or any(
        int(row['examples'])!=272115 or int(row['steps'])!=55 or int(row['short_batch_size'])!=2115 for row in epochs):
        raise ValueError('Incomplete epoch/batch coverage')
    if [int(row['epoch']) for row in validations]!=list(range(10,1001,10)):
        raise ValueError('Incorrect validation cadence')
    if set(d.read(path/'batch_order.json'))!={str(i) for i in range(1,1001)}:
        raise ValueError('Incomplete batch provenance')
    if not math.isclose(sum(int(row['epoch_time_ns'])/1e9 for row in epochs),state['training_seconds'],abs_tol=1e-7,rel_tol=0):
        raise ValueError('Training total differs from epoch records')
    best=max(validations,key=lambda row:float(row['overall_mrr']))
    for row in validations:
        epoch=int(row['epoch'])
        metrics=d.read(path/'validation'/f'epoch-{epoch:04d}.json')
        rank_metrics(metrics)
        if any(not math.isclose(float(row[f'{side}_{key}']),value,abs_tol=1e-12,rel_tol=0)
               for side in ('head','tail','overall') for key,value in metrics[side].items()):
            raise ValueError('Validation CSV differs from rank material')
    checkpoint=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
    last=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
    if int(best['epoch'])!=state['best_epoch'] or checkpoint['epoch']!=state['best_epoch'] or checkpoint['validation_mrr']!=state['best_validation_mrr']:
        raise ValueError('Checkpoint is not earliest validation maximum')
    if last['epoch']!=1000 or last['validation_mrr']!=float(validations[-1]['overall_mrr']):
        raise ValueError('Missing final checkpoint identity')
    return {'state':state,'path':outcome['path'],'best':best,'last':validations[-1],'validation':validations,
            'initial_weights':d.read(path/'initial_weights.json'),'batch_order':d.read(path/'batch_order.json')}


def verify_observations(root,records,record):
    from .mechanism_observe import sampling_summary
    from . import data
    dataset=data.load(root/'inputs/FB15K237')
    known=set(dataset['train'])
    train_keys=np.array(sorted((h*237+r)*14541+t for h,r,t in known),dtype=np.int64)
    report={'status':'passed','array_files':0,'fixed_parameter_cells':0,'historical_calls':0,
            'maximum_gradient_reconstruction_relative_error':0.0}
    if 'O-TENSOR' in records:
        result=records['O-TENSOR'];path=root/result['path']
        batches={item['index']:np.array(item['triples'],dtype=np.int64) for item in d.read(path/'batches.json')}
        expected=set(itertools.product(('initial-401','A-101','C-101'),GROUPS,(601,602,603),(0,1,2,3,54)))
        actual={(row['anchor'],row['group'],row['seed'],row['batch_index']) for row in result['observations']}
        if actual!=expected or len(actual)!=180:
            raise ValueError('Observation cell matrix mismatch')
        for row in result['observations']:
            with np.load(path/row['arrays'],allow_pickle=False) as arrays:
                positive=batches[row['batch_index']]
                direction=arrays['directions']
                replacement=arrays['replacement']
                negative=np.broadcast_to(positive[:,None,:],(*replacement.shape,3)).copy()
                negative[:,:,0]=np.where(direction,replacement,negative[:,:,0])
                negative[:,:,2]=np.where(~direction,replacement,negative[:,:,2])
                scores,weights=arrays['scores'],arrays['weights']
                logits=-record['config']['adv']*scores
                logits-=logits.max(1,keepdims=True)
                recalculated=np.exp(logits);recalculated/=recalculated.sum(1,keepdims=True)
                if not np.allclose(recalculated,weights,rtol=1e-5,atol=1e-7):
                    raise ValueError('Saved weights differ from detached score softmax')
                summary=sampling_summary(positive,negative,direction,scores,weights,14541,237)
                if summary!=row['summary']:
                    raise ValueError('Sampling summaries differ from raw observations')
                if row['group'] in ('M01','M11') and summary['duplicate_fraction']['max']!=0:
                    raise ValueError('Unique group produced duplicates')
                if np.any((negative==positive[:,None,:]).all(-1)):
                    raise ValueError('Observation contains original positive')
                keys=(negative[:,:,0]*237+negative[:,:,1])*14541+negative[:,:,2]
                indices=np.searchsorted(train_keys,keys)
                if np.any((indices<len(train_keys)) & (train_keys[np.minimum(indices,len(train_keys)-1)]==keys)):
                    raise ValueError('Observation contains a training-known true triple')
                head_count=direction.sum(1)
                valid=(head_count==75) if row['group'] in ('M00','M01') else ((head_count==0)|(head_count==150))
                if not np.all(valid):
                    raise ValueError('Observation direction policy mismatch')
            error=row['gradients']['gradient_reconstruction_relative_error']
            if not math.isfinite(error) or error>1e-4:
                raise ValueError('Invalid gradient reconstruction')
            report['maximum_gradient_reconstruction_relative_error']=max(error,report['maximum_gradient_reconstruction_relative_error'])
            report['fixed_parameter_cells']+=1
            report['array_files']+=1
    if 'O-A' in records:
        import random
        from .mechanism_observe import historical_observation
        result=records['O-A'];path=root/result['path']
        if {(row['seed'],row['batch_index']) for row in result['observations']}!=set(itertools.product((601,602,603),(0,1,2,3,54))):
            raise ValueError('Historical observation matrix mismatch')
        for row in result['observations']:
            with np.load(path/row['arrays'],allow_pickle=False) as arrays:
                if len(arrays['positive'])!=row['positive_count'] or arrays['negative'].shape!=(len(arrays['positive']),150,3):
                    raise ValueError('Historical observation array mismatch')
                value=row['seed']*100003+row['batch_index']*10
                random.seed(value);np.random.seed(value)
                import torch
                negative,statistics=historical_observation(torch.tensor(arrays['positive'],dtype=torch.long),14541,150,known)
                if not np.array_equal(negative.numpy(),arrays['negative']) or any(statistics[key]!=row[key] for key in statistics):
                    raise ValueError('Historical observation RNG replay mismatch')
            report['historical_calls']+=1
            report['array_files']+=1
    return report


def export_csv(path,rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def observation_groups(rows):
    groups={}
    metrics=('head_weight_mass','duplicate_fraction','position_ess','unique_ess','weight_entropy')
    for row in rows:
        groups.setdefault((row['anchor'],row['group']),[]).append(row)
    result=[]
    for (anchor,group),items in sorted(groups.items()):
        count=sum(item['summary']['positive_count'] for item in items)
        record={'anchor':anchor,'group':group,'cells':len(items),'positive_count':count}
        record.update({name:sum(item['summary'][name]['mean']*item['summary']['positive_count'] for item in items)/count for name in metrics})
        for name in ('head','tail','negative','total'):
            record[f'{name}_gradient_norm_per_positive']=mean(item['gradients']['gradient_norms'][name]/item['summary']['positive_count'] for item in items)
        for name in ('head/tail','counterfactual_negative/negative'):
            values=[item['gradients']['gradient_cosines'].get(name) for item in items]
            values=[value for value in values if value is not None]
            record[name+'_cosine']=mean(values) if values else None
        result.append(record)
    return result


def render(output,summary,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']='mukg-mechanism-v1'
    def save(figure,name):
        figure.tight_layout()
        figure.savefig(output/f'{name}.png',dpi=160)
        figure.savefig(output/f'{name}.svg',metadata={'Date':None})
        plt.close(figure)
    rows=[]
    for task_id,item in records.items():
        if not task_id.startswith('V-'):
            continue
        state=item['state']
        rows.append({'task':task_id,'group':state['group'],'seed':state['seed'],
                     'best_epoch':state['best_epoch'],'best_validation_mrr':float(item['best']['overall_mrr']),
                     'last_validation_mrr':float(item['last']['overall_mrr']),
                     'best_head_mrr':float(item['best']['head_mrr']),'best_tail_mrr':float(item['best']['tail_mrr']),
                     'training_hours':state['training_seconds']/3600,'validation_hours':state['validation_seconds']/3600,
                     'wall_hours':state['wall_seconds']/3600})
    export_csv(output/'validation_by_seed.csv',rows)
    contrast_rows=[]
    for scope in ('best','last'):
        section=summary[scope]
        if section['status']=='complete':
            for category in ('comparisons','effects'):
                for name,item in section[category].items():
                    contrast_rows.append({'scope':scope,'category':category,'comparison':name,
                                          'geometric_mean':item['geometric_mean'],'ci95_low':item['ci95'][0],
                                          'ci95_high':item['ci95'][1],'paired_values':str(item['paired_ratios'])})
    export_csv(output/'paired_effects.csv',contrast_rows)
    for field,label,name in (('epoch','Epoch','validation_vs_epoch'),
                             ('training_seconds','Training time (hours)','validation_vs_training_time'),
                             ('wall_seconds','Wall time (hours)','validation_vs_wall_time')):
        fig,axis=plt.subplots(figsize=(9,5))
        for group in GROUPS:
            series=[item['validation'] for key,item in records.items() if key.startswith('V-') and item['state']['group']==group]
            if not series:
                continue
            x=np.array([[float(row[field])/(1 if field=='epoch' else 3600) for row in values] for values in series])
            y=np.array([[float(row['overall_mrr']) for row in values] for values in series])
            line,=axis.plot(x.mean(0),y.mean(0),label=f'{group}, n={len(series)}')
            for xx,yy in zip(x,y):
                axis.plot(xx,yy,color=line.get_color(),alpha=.18,linewidth=.7)
        axis.set_xlabel(label);axis.set_ylabel('Bidirectional filtered validation MRR');axis.grid(alpha=.2)
        if axis.lines:axis.legend()
        save(fig,name)
    fig,axis=plt.subplots(figsize=(8,5))
    x=np.arange(len(GROUPS))
    for offset,side in ((-.2,'head'),(.2,'tail')):
        values=[mean(float(item['best'][f'{side}_mrr']) for key,item in records.items()
                     if key.startswith('V-') and item['state']['group']==group)
                if any(key.startswith('V-') and item['state']['group']==group for key,item in records.items()) else np.nan for group in GROUPS]
        axis.bar(x+offset,values,width=.38,label=side)
    axis.set_xticks(x,GROUPS);axis.set_ylabel('Best-checkpoint validation MRR');axis.legend()
    save(fig,'validation_by_direction')
    observation_rows=observation_groups(records.get('O-TENSOR',{}).get('observations',[]))
    export_csv(output/'mechanism_by_anchor.csv',observation_rows)
    if observation_rows:
        anchors=['initial-401','A-101','C-101']
        for field,label,name in (('head_weight_mass','Head negative weight mass','mechanism_weight_mass'),
                                 ('duplicate_fraction','Duplicate negative fraction','mechanism_duplicate_fraction')):
            fig,axes=plt.subplots(1,3,figsize=(12,4),squeeze=False)
            for axis,anchor in zip(axes[0],anchors):
                values={row['group']:row[field] for row in observation_rows if row['anchor']==anchor}
                axis.bar(GROUPS,[values.get(group,np.nan) for group in GROUPS])
                axis.set_title(anchor);axis.set_ylabel(label);axis.grid(axis='y',alpha=.2)
            save(fig,name)
        fig,axes=plt.subplots(1,3,figsize=(12,4),squeeze=False)
        for axis,anchor in zip(axes[0],anchors):
            values={row['group']:row for row in observation_rows if row['anchor']==anchor}
            for offset,side in ((-.2,'head'),(.2,'tail')):
                axis.bar(np.arange(4)+offset,[values.get(group,{}).get(f'{side}_gradient_norm_per_positive',np.nan) for group in GROUPS],width=.38,label=side)
            axis.set_xticks(np.arange(4),GROUPS);axis.set_title(anchor);axis.set_ylabel('Mean gradient norm / positive');axis.legend()
        save(fig,'mechanism_gradients')
    lines=['# muKG 机制实验报告','',f'状态为 {summary["status"]}，有效科学任务 {len(records)}/14。','',
           '四组使用GPU采样与训练，三个配对种子，固定1000轮，验证选模。指标来自验证集。','']
    if summary['best']['status']=='complete':
        lines.extend(['|组别|最佳验证MRR均值|末轮MRR均值|','|---|---:|---:|'])
        for group in GROUPS:
            lines.append(f'|{group}|{summary["best"]["group_means"][group]:.6f}|{summary["last"]["group_means"][group]:.6f}|')
        primary=summary['best']['comparisons']['M10/M00']
        lines.extend(['',f'预设主比较M10/M00为 {primary["geometric_mean"]:.6f}，95%区间 {primary["ci95"]}。',
                      f'按预设均值排序选择 {summary["best"]["candidate"]}。新增候选改善标记为 {summary["best"]["new_candidate_found"]}。'])
    lines.extend(['','## 机制观测','',
                  '固定参数的权重及梯度变化只说明这些锚点下的即时机制。方向干预改变候选组织和梯度相关性，softmax归一化的训练质量作用仍需独立损失对照。',
                  '原始观测见observations_by_cell.json，历史采样兜底及重复见historical_observations.json。',
                  'mechanism_by_anchor.csv按锚点与策略汇总。权重/重复/ESS按正样本数量加权，梯度范数先除以该batch正样本数再对观测单元求均值，属于描述性统计。',
                  '','## 解释边界','',
                  '主比较与探索性因素、交互均完整报告，三种子log-t区间依赖小样本方法假设。',
                  '候选按验证集选择，结果不能替代最终测试质量确认，也不能直接证明相对历史A满足0.95界限。',
                  '此前已查看FB15k-237测试结果，此阶段为结果驱动的探索。历史C/A测试MRR比0.922624及其未达标结论继续保留。',
                  '历史A存在重试换向、轮内无放回和末轮兜底，当前两个因素不能覆盖全部差异。',
                  '全部组在GPU执行，训练任务耗时用于描述当前路径，不据此声称新候选匹配CPU/GPU加速。',
                  '固定预算不等于充分收敛，所有有效负结果保留。',''])
    (output/'REPORT.md').write_text('\n'.join(lines))


def analyze(root,output):
    from . import mechanism as m
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists() or ((root/'artifacts.json').exists() and output.is_relative_to(root)) or any(output.is_relative_to(root/name) for name in ('source','inputs','preparation','jobs','documents')):
        raise FileExistsError('Analysis output exists or overlaps frozen inputs')
    record=m.read_freeze(root)
    if d.source_identity(root/'inputs/FB15K237')!=record['provenance']:
        raise ValueError('Use the frozen source version for analysis')
    outcomes=d.read(root/'outcomes.json') if (root/'outcomes.json').exists() else {}
    records={};missing={};best={};last={};pairs={}
    for task in m.task_catalog():
        outcome=outcomes.get(task['id'],{'status':'pending'})
        if outcome['status']!='complete':
            missing[task['id']]=outcome
            continue
        item=verify_job(root,task,outcome,record)
        records[task['id']]=item
        if task['kind']=='validation':
            seed=str(task['seed'])
            identity=(item['initial_weights'],item['batch_order'])
            if seed in pairs and pairs[seed]!=identity:
                raise ValueError('Paired initialization/positive order mismatch')
            pairs[seed]=identity
            best.setdefault(seed,{})[task['group']]=float(item['best']['overall_mrr'])
            last.setdefault(seed,{})[task['group']]=float(item['last']['overall_mrr'])
    observed=verify_observations(root,records,record)
    summary={'status':'complete' if not missing else 'incomplete','protocol':m.PROTOCOL,
             'freeze_sha256':record['freeze_sha256'],'missing':missing,
             'best':matrix_statistics(best),'last':matrix_statistics(last),
             'verification':{**observed,'valid_tasks':len(records),'validation_rank_files':100*sum(key.startswith('V-') for key in records),
                             'paired_initialization_and_orders_verified':True}}
    if missing:
        summary['best']['candidate']=None
        summary['best']['new_candidate_found']=None
    output.mkdir(parents=True)
    d.write(output/'summary.json',summary)
    d.write(output/'observations_by_cell.json',records.get('O-TENSOR',{}).get('observations',[]))
    d.write(output/'historical_observations.json',records.get('O-A',{}).get('observations',[]))
    render(output,summary,records)
    d.seal(output)
    return summary
