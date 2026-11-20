"""Evidence-only diagnostic analysis; no training/test evaluation entry point."""
import csv
import json
import math
from pathlib import Path
from . import diagnostics as d


def budget_rows(rows):
    per={}
    for row in rows:
        train=row['training_seconds']/row['epochs_completed']
        validation=row['validation_seconds']/row['validation_count']
        overhead=max(0,row['wall_seconds']-row['training_seconds']-row['validation_seconds'])
        per[row['group']]={'observed_train_epoch_seconds':train,'observed_validation_seconds':validation,
            'observed_other_wall_seconds':overhead,'projected_training_seconds':1000*train,
            'projected_validation_seconds':100*validation,
            'projected_test_seconds':validation*20466/17535,
            'projected_other_wall_seconds':overhead}
        per[row['group']]['quality_job_seconds']=sum(per[row['group']][key] for key in (
            'projected_training_seconds','projected_validation_seconds','projected_test_seconds','projected_other_wall_seconds'))
        per[row['group']]['throughput_job_lower_projection_seconds']=6*train+overhead
    quality=3*sum(row['quality_job_seconds'] for row in per.values())
    throughput=5*sum(row['throughput_job_lower_projection_seconds'] for row in per.values())
    return {'candidate_epochs':1000,'formal_epochs_frozen':False,'per_group':per,
            'quality_nine_jobs_hours':quality/3600,
            'throughput_fifteen_jobs_lower_projection_hours':throughput/3600,
            'known_projection_hours_with_reserve':1.3*(quality+throughput)/3600,
            'trace_seconds':'unknown: synchronized instrumentation not measured',
            'reserve_fraction':.30,
            'assumptions':['linear scaling of observed 100 epochs; no early stopping',
                '100 full validations at candidate 1000 epochs',
                'test time extrapolated by query count, not measured or evaluated',
                'throughput: one warmup and five measurement epochs; startup/save cost approximate',
                'trace duration and additional checks unknown; sum is not a complete total budget']}


def csv_rows(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def analyze(root):
    root=Path(root);catalog=d.read(root/'catalog.json');d.validate_binding(root,catalog)
    outcomes=d.read(root/'outcomes.json');records=[];missing=[]
    for task in d.TASKS:
        state=outcomes.get(task)
        if not state or state['status']!='complete':missing.append({'task':task,'reason':state or 'not run'});continue
        path=root/state['path']
        if d.sha(path/'artifacts.json')!=state['artifacts_sha256']:raise ValueError('Job artifact anchor changed')
        actual=d.verify_job(root,task,path,catalog)
        vals=csv_rows(path/'validation.csv')
        best=max(vals,key=lambda row:float(row['mrr']))
        last=vals[-1]
        if int(last['epoch'])!=100 or int(best['epoch'])!=actual['best_epoch']:raise ValueError('Best/last checkpoint summary mismatch')
        if any(not math.isfinite(float(row[key])) for row in vals for key in ('mrr','head_mrr','tail_mrr')):raise ValueError('Nonfinite validation metric')
        records.append({'group':task.split('-')[1],'task':task,'path':state['path'],**{key:actual[key] for key in ('training_seconds','validation_seconds','wall_seconds','epochs_completed','validation_count')},
            'best_epoch':int(best['epoch']),'best_mrr':float(best['mrr']),'best_head_mrr':float(best['head_mrr']),'best_tail_mrr':float(best['tail_mrr']),
            'last_mrr':float(last['mrr']),'last_head_mrr':float(last['head_mrr']),'last_tail_mrr':float(last['tail_mrr']),
            'last_three_validation_mrr':[float(row['mrr']) for row in vals[-3:]],
            'peak_allocated_bytes':max(int(row['peak_allocated_bytes']) for row in csv_rows(path/'epochs.csv')),
            'peak_reserved_bytes':max(int(row['peak_reserved_bytes']) for row in csv_rows(path/'epochs.csv')),
            'initial_weights':d.read(path/'initial_weights.json'),'batch_order':d.read(path/'batch_order.json')})
    if records:
        for record in records[1:]:
            if record['initial_weights']!=records[0]['initial_weights'] or record['batch_order']!=records[0]['batch_order']:
                raise ValueError('Paired initial weights or positive batch order differs')
    output=root/'analysis';output.mkdir(exist_ok=True)
    complete=len(records)==3 and not missing
    summary={'complete':complete,'verdict':'DESCRIPTIVE' if complete else 'INCOMPLETE',
        'formal_execution_allowed':False,'test_evaluated':False,'formal_epochs_frozen':False,
        'records':[{k:v for k,v in row.items() if k not in ('initial_weights','batch_order')} for row in records],
        'missing':missing,'catalog_sha256':d.sha(root/'catalog.json')}
    if complete:summary['budget']=budget_rows(records)
    d.write(output/'summary.json',summary)
    if records:
        fields=[key for key in summary['records'][0] if key!='last_three_validation_mrr']
        with (output/'diagnostic_summary.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            writer.writerows({key:row[key] for key in fields} for row in summary['records'])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for x,label,name in [('training_seconds','Cumulative training time (s)','validation_vs_training_time.png'),('wall_seconds','Wall time since task start (s)','validation_vs_wall_time.png')]:
            fig,ax=plt.subplots()
            for row in records:
                vals=csv_rows(root/row['path']/'validation.csv')
                ax.plot([float(v[x]) for v in vals],[float(v['mrr']) for v in vals],marker='.',label=row['group'])
            ax.set(xlabel=label,ylabel='Validation bidirectional filtered MRR');ax.legend();fig.tight_layout();fig.savefig(output/name,dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for side,ax in zip(('head','tail'),axes):
            for row in records:
                vals=csv_rows(root/row['path']/'validation.csv')
                ax.plot([int(v['epoch']) for v in vals],[float(v[f'{side}_mrr']) for v in vals],label=row['group'])
            ax.set(xlabel='Epoch',ylabel=f'Validation {side} MRR');ax.legend()
        fig.tight_layout();fig.savefig(output/'head_tail_by_epoch.png',dpi=160);plt.close(fig)
    lines=['# A/B/C 100 epoch 诊断报告','',f'状态：{summary["verdict"]}。仅seed7与验证集，不进行test评估，不冻结正式轮数。','',
        '|组别|训练秒|验证秒|墙钟秒|最佳epoch|最佳MRR|最佳头MRR|最佳尾MRR|末轮MRR|末轮头MRR|末轮尾MRR|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        lines.append('|'+ '|'.join([row['group'],*[f'{row[k]:.2f}' for k in ('training_seconds','validation_seconds','wall_seconds')],str(row['best_epoch']),*[f'{row[k]:.6f}' for k in ('best_mrr','best_head_mrr','best_tail_mrr','last_mrr','last_head_mrr','last_tail_mrr')]])+'|')
    lines+=['','## 证据边界','',
        '- 最佳值在各自验证选择的epoch；末轮另列，不能混为同轮比较。',
        '- 单种子、100epoch诊断；不作正式加速、非劣效或收敛判断，不追加有利种子。',
        '- 峰值显存属于整轮训练路径；边界遥测可能遗漏短暂干扰。',
        '- A/B同时改变规则与CPU实现，B/C包含设备与传输方式；不做单规则因果归因。',
        '- 原始排名、每轮完整头尾指标、初始化及正样本顺序证据保存在各任务目录。']
    if complete:
        b=summary['budget']
        lines+=['','## 1000 epoch候选与预算','',
            f'九个质量任务线性外推约 {b["quality_nine_jobs_hours"]:.2f} 小时；15个吞吐任务下界式估计约 {b["throughput_fifteen_jobs_lower_projection_hours"]:.2f} 小时。',
            f'上述已估部分加30%余量约 {b["known_projection_hours_with_reserve"]:.2f} 小时；尚不含未知trace与额外检查，不能称完整预算。',
            'test仅按验证查询数量比例外推，未实际访问或评估。B已纳入估算，详见summary.json。','',
            '建议：将1000epoch保留为待讨论的统一固定预算；仅100epoch无法证明训练充分性。若末段仍改善，明确风险；不得由诊断自动推导需加跑的轮数。',
            '本轮到此停止，等待用户确认正式轮数及后续执行安排。']
    else:lines+=['','## 缺失任务','',json.dumps(missing,ensure_ascii=False,indent=2),'','证据不完整，不推荐冻结正式运行。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary
