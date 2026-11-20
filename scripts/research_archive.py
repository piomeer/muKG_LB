"""Local lossless research snapshot; no training, networking or secret capture."""
import argparse,csv,hashlib,json,os,shutil
from pathlib import Path
EXCLUDED={'.git','.aws','.codex','.agents','.idea','.vscode','.superpowers','.research-archive','.worktrees','node_modules','__pycache__','.pytest_cache','venv','.venv','env','build'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def files(root):
 for parent,dirs,names in os.walk(root,followlinks=False):
  dirs[:]=sorted(d for d in dirs if d not in EXCLUDED)
  for d in dirs:
   if Path(parent,d).is_symlink():raise ValueError(f'Explicit review required for symlink: {Path(parent,d)}')
  for n in sorted(names):
   p=Path(parent,n)
   if n=='.env' or n.startswith('.env.') or n.endswith(('.pyc','.pyo')):continue
   if p.is_symlink():raise ValueError(f'Explicit review required for symlink: {p}')
   yield p

def snapshot(root,dest):
 root,dest=Path(root).resolve(),Path(dest).resolve()
 dest.mkdir(parents=True,exist_ok=False); target=dest/'snapshot';target.mkdir()
 rows=[]
 for p in files(root):
  if dest==p or dest in p.parents:continue
  rel=p.relative_to(root); out=target/rel;out.parent.mkdir(parents=True,exist_ok=True)
  before=p.stat(); digest=sha(p);shutil.copy2(p,out);after=p.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or sha(out)!=digest:raise RuntimeError(f'Source changed during snapshot: {p}')
  rows.append({'path':rel.as_posix(),'bytes':out.stat().st_size,'sha256':digest})
 (dest/'manifest.json').write_text(json.dumps({'source':str(root),'exclusions':sorted(EXCLUDED),'files':rows},indent=2)+'\n')
 return rows

def verify(dest):
 dest=Path(dest);data=json.loads((dest/'manifest.json').read_text());errors=[]
 expected=set()
 for row in data['files']:
  rel=Path(row['path'])
  if rel.is_absolute() or '..' in rel.parts or str(rel) in expected:raise ValueError('Unsafe or duplicate archive path')
  expected.add(rel.as_posix());p=dest/'snapshot'/rel
  if p.is_symlink() or not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:errors.append(row['path'])
 actual={p.relative_to(dest/'snapshot').as_posix() for p in (dest/'snapshot').rglob('*') if p.is_file() or p.is_symlink()}
 errors.extend(sorted(actual^expected))
 if (dest/'seal.json').exists():
  hashes=json.loads((dest/'seal.json').read_text())['metadata_hashes']
  actual={p.relative_to(dest).as_posix() for p in dest.rglob('*') if (p.is_file() or p.is_symlink()) and 'snapshot'!=p.relative_to(dest).parts[0] and p.name!='seal.json'}
  errors.extend(sorted(actual^set(hashes)))
  for name,digest in hashes.items():
   rel=Path(name)
   if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe metadata path')
   p=dest/rel
   if p.is_symlink() or not p.is_file() or sha(p)!=digest:errors.append(name)
 return sorted(set(errors))

def restore(dest,target):
 if verify(dest):raise ValueError('Archive verification failed')
 shutil.copytree(Path(dest)/'snapshot',target)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['snapshot','verify','restore']);p.add_argument('source',type=Path);p.add_argument('target',type=Path,nargs='?');a=p.parse_args()
 if a.action=='snapshot':print(json.dumps({'files':len(snapshot(a.source,a.target))}))
 elif a.action=='verify':
  errors=verify(a.source);print(json.dumps({'errors':errors}));raise SystemExit(bool(errors))
 else:restore(a.source,a.target)
