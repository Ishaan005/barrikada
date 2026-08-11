"""Build hook that places the registration bootstrap at site-packages root."""

from pathlib import Path
from shutil import copyfile

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithAutoload(build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent / "src" / "barrikade_jentic_autoload.pth"
        copyfile(source, Path(self.build_lib) / source.name)


setup(cmdclass={"build_py": BuildWithAutoload})
