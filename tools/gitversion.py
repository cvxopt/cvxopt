#!/usr/bin/env python3
"""Get the project version from pyproject.toml, with git metadata for dev builds."""
import os
import subprocess


def get_base_version():
    """Read the base version from pyproject.toml."""
    toml_path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
    with open(toml_path) as f:
        for line in f:
            if line.startswith('version ='):
                return line.split('=', 1)[1].strip().strip('"').strip("'")
    raise RuntimeError('version not found in pyproject.toml')


def git_version(version):
    """Append git date and hash to dev versions."""
    git_hash = ''
    try:
        p = subprocess.Popen(
            ['git', 'log', '-1', '--format="%H %aI"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__),
        )
    except FileNotFoundError:
        pass
    else:
        out, err = p.communicate()
        if p.returncode == 0:
            git_hash, git_date = (
                out.decode('utf-8')
                .strip()
                .replace('"', '')
                .split('T')[0]
                .replace('-', '')
                .split()
            )
            if 'dev' in version:
                version += f'+git{git_date}.{git_hash[:7]}'

    return version, git_hash


if __name__ == '__main__':
    version, _ = git_version(get_base_version())
    print(version)
