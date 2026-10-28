import tempfile
import unittest
from pathlib import Path

from shadowdream.snapshot import snapshot_id


class SnapshotTest(unittest.TestCase):
    def test_local_base_and_adapter_changes_change_identity(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root) / 'base'
            adapter = Path(root) / 'adapter'
            base.mkdir()
            adapter.mkdir()
            (base / 'weights').write_text('one')
            (adapter / 'weights').write_text('a')
            config = {'model': {'path': str(base)}, 'service_adapter': str(adapter)}
            old = snapshot_id(None, config)
            (adapter / 'weights').write_text('b')
            self.assertNotEqual(old, snapshot_id(None, config))
            old = snapshot_id(None, config)
            (base / 'weights').write_text('two')
            self.assertNotEqual(old, snapshot_id(None, config))
