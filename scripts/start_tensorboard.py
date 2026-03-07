#!/usr/bin/env python3
"""TensorBoard startup script with pkg_resources compatibility fix."""
import sys
import importlib_metadata

class MockEntryPoint:
    """Mock EntryPoint with resolve method."""
    def __init__(self, ep):
        self.ep = ep
        self.name = ep.name
        self.group = ep.group
        self.value = ep.value
        self.dist = ep.dist
    
    def resolve(self):
        return self.ep.load()
    
    def __getattr__(self, name):
        return getattr(self.ep, name)

class MockPkgResources:
    """Mock pkg_resources for TensorBoard compatibility."""
    
    def iter_entry_points(self, group, name=None):
        eps = importlib_metadata.entry_points()
        if hasattr(eps, 'select'):
            selected = eps.select(group=group, name=name) if name else eps.select(group=group)
        else:
            selected = [ep for ep in eps if ep.group == group and (name is None or ep.name == name)]
        
        # Wrap entry points with mock that has resolve method
        return [MockEntryPoint(ep) for ep in selected]

# Replace pkg_resources with our mock
sys.modules['pkg_resources'] = MockPkgResources()

# Import and start TensorBoard
from tensorboard import program

if __name__ == "__main__":
    import subprocess
    subprocess.run([
        'tensorboard', 'serve',
        '--logdir', '/home/ubuntu/train_QLoRA/data/ai-lab/logs',
        '--host', '0.0.0.0',
        '--port', '6006'
    ])
