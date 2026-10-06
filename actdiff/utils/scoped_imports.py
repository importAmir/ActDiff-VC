"""Import hook: while a module inside a third-party repo is importing, its bare top-level
imports (e.g. `import utils`) resolve inside that repo first, so repos cannot shadow each other."""
import importlib.machinery
import importlib.util
import sys
from pathlib import Path


class ScopedTopLevelFinder:
    def __init__(self, repo_roots):
        self.repo_roots = [Path(root).resolve() for root in repo_roots.values()]

    def Active_Repo_Root(self):
        """Repo root containing the module that is currently importing, if any."""
        frame = sys._getframe(1)
        while frame is not None:
            module_file = frame.f_globals.get('__file__')
            if module_file:
                module_path = Path(module_file).resolve()
                for root in self.repo_roots:
                    if module_path.is_relative_to(root):
                        return root
            frame = frame.f_back
        return None

    def find_spec(self, fullname, path, target=None):  # importlib finder protocol
        if '.' in fullname:
            return None
        root = self.Active_Repo_Root()
        if root is None:
            return None

        package_dir = root / fullname
        if package_dir.is_dir():
            init_file = package_dir / '__init__.py'
            if init_file.exists():
                return importlib.util.spec_from_file_location(
                    fullname,
                    init_file.as_posix(),
                    loader=importlib.machinery.SourceFileLoader(fullname, init_file.as_posix()),
                    submodule_search_locations=[package_dir.as_posix()],
                )
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [package_dir.as_posix()]
            return spec

        for ext in ('.py', '.so', '.pyd'):
            candidate = root / f'{fullname}{ext}'
            if candidate.exists():
                return importlib.util.spec_from_file_location(
                    fullname,
                    candidate.as_posix(),
                    loader=importlib.machinery.SourceFileLoader(fullname, candidate.as_posix()),
                )
        return None


def Install_Scoped_Imports(repo_roots):
    sys.meta_path.insert(0, ScopedTopLevelFinder(repo_roots))
