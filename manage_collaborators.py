"""
Helper script to add B.Tech Final Year Project group members as collaborators
to the shared GitHub repository so all members can push commits and trigger
the shared deployment automatically.

Usage:
    .\.venv\Scripts\python.exe manage_collaborators.py add <github_username_1> <github_username_2> ...
    .\.venv\Scripts\python.exe manage_collaborators.py list
"""

import json
import subprocess
import sys
import urllib.request
import urllib.error


REPO_OWNER = "Yadul12"
REPO_NAME = "learned-cost-estimation"


def get_github_token() -> str:
    proc = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n",
        text=True,
        capture_output=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("No GitHub token found in git credential store.")


def add_collaborator(username: str, permission: str = "push") -> None:
    token = get_github_token()
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/collaborators/{username}"
    data = json.dumps({"permission": permission}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "Antigravity-Group-Manager",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"[SUCCESS] Invited/Added '{username}' with '{permission}' access (HTTP {resp.status}).")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"[ERROR] Failed to add '{username}': HTTP {e.code} - {body}")


def list_collaborators() -> None:
    token = get_github_token()
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/collaborators"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "Antigravity-Group-Manager",
        },
    )
    with urllib.request.urlopen(req) as resp:
        users = json.loads(resp.read().decode("utf-8"))
        print("Current Repository Collaborators:")
        for u in users:
            print(f"  - {u['login']} ({u['html_url']}) | Permissions: {u.get('permissions', {})}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1].lower()
    if cmd == "list":
        list_collaborators()
    elif cmd == "add":
        for user in sys.argv[2:]:
            add_collaborator(user, permission="maintain")
    else:
        print(__doc__)
