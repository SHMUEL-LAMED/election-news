#!/usr/bin/env python3
"""Keep the last video-upload runs in one GitHub issue, so a failure can be read without the Actions log.

The issue body is edited in place (no comment per run), newest run first.
"""
import collections
import datetime as dt
import json
import os
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

TITLE = 'יומן ריצות כתבות וידאו'
MARK = '<!-- video-run -->'
KEEP = 30


def summary(lines):
    """Count the script's own result codes: 'video_download_failed mako blur_provider_429' and the like."""
    counts = collections.Counter()
    for line in lines:
        words = line.split()
        if not words or not re.fullmatch(r'[a-z_]+:?', words[0]):
            continue
        if words[0] in ('video_source_failed', 'video_download_failed', 'post_blur_deferred', 'post_blur_queue_failed'):
            counts[' '.join(words[:4])] += 1
        elif words[0] in ('video_ready', 'video_archived', 'video_staged', 'video_emailed', 'post_blur_replaced', 'blur_processing_retry'):
            counts[words[0]] += 1
    return counts


def entry(status, event, run, lines, now=None):
    now = now or dt.datetime.now(ZoneInfo('Asia/Jerusalem'))
    counts = summary(lines)
    head = '%s **%s** · %s · %s · run %s' % (MARK, status, now.strftime('%Y-%m-%d %H:%M'), event.strip(), run)
    body = ['- %d × `%s`' % (n, key) for key, n in counts.most_common(12)]
    # The last lines carry the final verdict or the Python error.
    tail = [x[:300].replace('`', "'") for x in lines if x.strip()][-12:]
    if not lines:
        tail = ['no script log: the job stopped before the upload step']
    return '\n'.join([head, *body, '```', *tail, '```'])


def merged(new, old):
    parts = [x for x in old.split(MARK) if x.strip()] if MARK in old else []
    return '\n\n'.join([new] + [MARK + x.rstrip() for x in parts[:KEEP - 1]])[:60000]


def gh(*args, text=None):
    return subprocess.run(['gh', *args], input=text, capture_output=True, text=True, check=True, timeout=40).stdout


def main():
    path = sys.argv[1]
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines() if os.path.exists(path) else []
    repo = os.environ['GITHUB_REPOSITORY']
    new = entry(os.environ.get('JOB_STATUS', '?'), os.environ.get('RUN_EVENT', ''), os.environ.get('GITHUB_RUN_ID', ''), lines)
    found = json.loads(gh('issue', 'list', '--repo', repo, '--state', 'open', '--search', 'in:title "%s"' % TITLE, '--json', 'number,title,body'))
    issue = next((x for x in found if x['title'] == TITLE), None)
    if issue:
        gh('issue', 'edit', str(issue['number']), '--repo', repo, '--body-file', '-', text=merged(new, issue['body'] or ''))
    else:
        gh('issue', 'create', '--repo', repo, '--title', TITLE, '--body-file', '-', text=new)
    print('video_run_reported', flush=True)


if __name__ == '__main__':
    main()
