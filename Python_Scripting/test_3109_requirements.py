"""3.10.9: no dependency may upgrade torch in either environment.

Seen on the first install of the coqui-tts environment: speechbrain 1.1.2,
allowed by "speechbrain>=1.0", requires torch >= 2.7.1. pip upgraded torch to
satisfy it, past 2.9, and coqui-tts then stopped on import asking for
torchcodec. requirements.txt had the same opening: a fresh TTS 0.22 install
would have upgraded torch too, and TTS 0.22 cannot load XTTS with torch >= 2.6.

Both files now hold torch and torchaudio at 2.5.1 (a build such as
2.5.1+cu121, installed first, matches) and keep speechbrain below 1.1.

    python -m unittest test_3109_requirements -v
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except ImportError:                      # packaging comes with pip and setuptools
    Requirement = Version = None


def requirements(name):
    """{project name (lower case): specifier text} of a requirements file."""
    out = {}
    for line in (ROOT / name).read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            m = re.match(r'([A-Za-z0-9_.\-]+)\s*(.*)$', line)
            out[m.group(1).lower()] = m.group(2).replace(' ', '')
    return out


class RequirementFileTests(unittest.TestCase):
    FILES = ('requirements.txt', 'requirements-coqui-tts.txt')

    def test_torch_is_held_in_both(self):
        for name in self.FILES:
            with self.subTest(file=name):
                reqs = requirements(name)
                self.assertEqual(reqs.get('torch'), '==2.5.1')
                self.assertEqual(reqs.get('torchaudio'), '==2.5.1')

    def test_speechbrain_stays_below_1_1(self):
        for name in self.FILES:
            with self.subTest(file=name):
                self.assertEqual(requirements(name).get('speechbrain'), '>=1.0,<1.1')

    @unittest.skipIf(Requirement is None, 'packaging is not installed')
    def test_every_line_parses_and_a_local_torch_build_matches(self):
        for name in self.FILES:
            for project, spec in requirements(name).items():
                with self.subTest(file=name, project=project):
                    Requirement(project + spec)
        torch = Requirement('torch==2.5.1')
        for installed in ('2.5.1', '2.5.1+cu121', '2.5.1+cpu'):
            self.assertIn(Version(installed), torch.specifier)
        for newer in ('2.7.1', '2.9.0'):
            self.assertNotIn(Version(newer), torch.specifier)
        self.assertNotIn(Version('1.1.2'), Requirement('speechbrain>=1.0,<1.1').specifier)
        self.assertIn(Version('1.0.3'), Requirement('speechbrain>=1.0,<1.1').specifier)

    def test_the_installer_puts_torch_first(self):
        script = (ROOT / 'install.sh').read_text(encoding='utf-8')
        first_torch = script.index('pip install --quiet "torch==2.5.1" "torchaudio==2.5.1"')
        requirements_step = script.index('pip install --quiet -r "$HERE/$REQUIREMENTS"')
        self.assertLess(first_torch, requirements_step)


if __name__ == '__main__':
    unittest.main(verbosity=2)
