"""Identity of a base model and adapter used during evaluation."""
import hashlib
import json
from pathlib import Path


def _digest_directory(path):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f'model snapshot is not a directory: {path}')
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob('*') if p.is_file() and not p.name.startswith('.'))
    if not files:
        raise ValueError(f'model snapshot directory is empty: {path}')
    for file in files:
        relative = file.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, 'big'))
        digest.update(relative)
        with file.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def snapshot_id(model, config):
    """Hash actual local model/adapter content; remote models use resolved revision."""
    source = config['model']['path']
    local = Path(source).expanduser()
    if local.exists():
        base = {'local_sha256': _digest_directory(local)}
    else:
        commit = getattr(getattr(model.model, 'config', None), '_commit_hash', None)
        if not commit:
            raise ValueError('remote model must expose a resolved immutable revision')
        base = {'repository': source, 'commit': commit}
    adapter = config.get('service_adapter')
    identity = {'base': base, 'adapter_sha256': _digest_directory(adapter) if adapter else None}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
