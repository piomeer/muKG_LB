"""Closed local snapshot with streamed content verification; no extraction."""
import hashlib
from pathlib import Path
import tarfile
from . import diagnostics as d


def archive_run(root,destination):
    root=Path(root).resolve();destination=Path(destination).resolve()
    if root==destination or root in destination.parents:raise ValueError('Archive must be outside run directory')
    destination.mkdir(parents=True,exist_ok=False)
    d.seal(root)
    inventory=d.read(root/'artifacts.json')
    expected={**inventory,'artifacts.json':{'sha256':d.sha(root/'artifacts.json'),'bytes':(root/'artifacts.json').stat().st_size}}
    archive=destination/'diagnostic-source-results.tar.gz'
    with tarfile.open(archive,'w:gz') as bundle:
        for name in sorted(expected):bundle.add(root/name,arcname=name,recursive=False)
    errors=[];seen=set()
    with tarfile.open(archive,'r|gz') as bundle:
        for member in bundle:
            if member.name in seen or member.name not in expected or not member.isfile():
                errors.append('unexpected/duplicate member: '+member.name);continue
            seen.add(member.name);digest=hashlib.sha256()
            stream=bundle.extractfile(member)
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            metadata=expected[member.name]
            if member.size!=metadata['bytes'] or digest.hexdigest()!=metadata['sha256']:errors.append(member.name)
    if seen!=set(expected):errors.append('missing archive members')
    receipt={'archive':str(archive),'sha256':d.sha(archive),'bytes':archive.stat().st_size,
        'files':len(expected),'source_run':str(root),'verification_errors':errors,
        'scope':'source/config/environment/diagnostic results/report; official dataset hashes reference existing historical archive; same-host recovery only'}
    d.write(destination/'receipt.json',receipt)
    if errors:raise ValueError(f'Archive verification failed: {errors}')
    d.verify_seal(root)
    return receipt
