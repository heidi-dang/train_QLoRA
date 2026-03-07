#!/usr/bin/env python3
"""Repo scraper: clone/update repositories and collect source files.

Writes clones to ai-lab/repos/<owner_repo>/ and a filelist to ai-lab/datasets/repos_filelist.txt
"""
import os
import subprocess
import sys
import logging
from urllib.parse import urlparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AI_LAB = os.path.join(ROOT, 'data', 'ai-lab')
REPOS_DIR = os.path.join(AI_LAB, 'repos')
REPO_LIST = os.path.join(AI_LAB, 'datasets', 'repos_premium.txt')
FILELIST_OUT = os.path.join(AI_LAB, 'datasets', 'repos_filelist.txt')

INCLUDE_EXT = {'.py', '.cpp', '.cc', '.h', '.hpp', '.c', '.rs', '.go'}
IGNORE_DIRS = {'tests', 'test', 'build', 'node_modules', '.git', 'third_party'}

logging.basicConfig(level=logging.INFO)


def read_repo_list(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
    return lines


def owner_repo_from_url(url):
    p = urlparse(url)
    path = p.path.rstrip('/').lstrip('/')
    return path.replace('.git', '').replace('/', '_')


def git_clone_or_update(url, dest):
    if os.path.exists(dest):
        logging.info('Updating %s', dest)
        try:
            subprocess.check_call(['git', '-C', dest, 'fetch', '--all'], stdout=subprocess.DEVNULL)
            subprocess.check_call(['git', '-C', dest, 'reset', '--hard', 'origin/HEAD'], stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            logging.exception('Failed to update %s', dest)
    else:
        logging.info('Cloning %s to %s', url, dest)
        try:
            subprocess.check_call(['git', 'clone', '--depth', '1', url, dest], stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            logging.exception('Failed to clone %s', url)


def collect_files(repo_path):
    matches = []
    for root, dirs, files in os.walk(repo_path):
        # prune ignored dirs
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fn in files:
            _, ext = os.path.splitext(fn)
            if ext.lower() in INCLUDE_EXT:
                matches.append(os.path.join(root, fn))
    return matches


def main():
    os.makedirs(REPOS_DIR, exist_ok=True)
    repos = read_repo_list(REPO_LIST)
    with open(FILELIST_OUT, 'w') as out:
        for url in repos:
            name = owner_repo_from_url(url)
            dest = os.path.join(REPOS_DIR, name)
            git_clone_or_update(url, dest)
            files = collect_files(dest)
            for f in files:
                out.write(f + '\n')
    logging.info('Wrote filelist to %s', FILELIST_OUT)


if __name__ == '__main__':
    main()
