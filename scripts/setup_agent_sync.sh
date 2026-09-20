#!/usr/bin/env bash
# Set up unattended GitHub sync + scheduled pressure-test runs on this machine.
#
#   bash scripts/setup_agent_sync.sh            # interactive, does nothing destructive
#   bash scripts/setup_agent_sync.sh --schedule # also install the systemd timer
#
# What it does:
#   1. Generates an ed25519 deploy key for THIS machine (if absent).
#   2. Prints the public key and waits for you to add it to the repo as a
#      read/write deploy key.
#   3. Switches `origin` to SSH and verifies the connection.
#   4. Installs a pre-push hook that runs the quick pressure test.
#   5. Optionally installs a systemd user timer for a nightly full run that
#      commits the report to a `pressure-test-reports` branch.
#
# Why a deploy key and not a personal access token: the key is scoped to this
# one repository and this one machine, lives only in ~/.ssh with 600 perms,
# never appears in a shell history or a git remote URL, and can be revoked from
# the repo's settings without touching your account. A PAT in a remote URL is
# account-wide and ends up in .git/config in plaintext.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY_PATH="${HOME}/.ssh/id_ed25519_hippo_$(hostname -s)"
SSH_HOST_ALIAS="github-hippo"
GH_SLUG="${GH_SLUG:-genevievewager/hippo}"
DO_SCHEDULE=0
[[ "${1:-}" == "--schedule" ]] && DO_SCHEDULE=1

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }

# ---------------------------------------------------------------- 1. key ----
say "1. Deploy key"
if [[ -f "${KEY_PATH}" ]]; then
  note "Reusing existing key: ${KEY_PATH}"
else
  ssh-keygen -t ed25519 -N '' -C "hippo-agent@$(hostname -s)" -f "${KEY_PATH}"
  note "Created ${KEY_PATH}"
fi
chmod 600 "${KEY_PATH}"
chmod 644 "${KEY_PATH}.pub"

# ------------------------------------------------------------ 2. ssh cfg ----
say "2. SSH config"
mkdir -p "${HOME}/.ssh"
touch "${HOME}/.ssh/config"
chmod 600 "${HOME}/.ssh/config"
if grep -q "Host ${SSH_HOST_ALIAS}\$" "${HOME}/.ssh/config"; then
  note "Host alias '${SSH_HOST_ALIAS}' already configured."
else
  cat >> "${HOME}/.ssh/config" <<EOF

Host ${SSH_HOST_ALIAS}
  HostName github.com
  User git
  IdentityFile ${KEY_PATH}
  IdentitiesOnly yes
EOF
  note "Added host alias '${SSH_HOST_ALIAS}'."
fi

say "Add this PUBLIC key as a deploy key with write access:"
echo
echo "    https://github.com/${GH_SLUG}/settings/keys/new"
echo
cat "${KEY_PATH}.pub"
echo
read -r -p "Press Enter once the deploy key is added (Ctrl-C to stop here)... " _

# --------------------------------------------------------------- 3. remote --
say "3. Remote"
cd "${REPO_ROOT}"
CURRENT="$(git remote get-url origin 2>/dev/null || echo '')"
note "current origin: ${CURRENT:-<none>}"
if [[ "${CURRENT}" != "${SSH_HOST_ALIAS}:${GH_SLUG}.git" ]]; then
  if [[ -n "${CURRENT}" ]]; then
    git remote set-url origin "${SSH_HOST_ALIAS}:${GH_SLUG}.git"
  else
    git remote add origin "${SSH_HOST_ALIAS}:${GH_SLUG}.git"
  fi
  note "origin -> ${SSH_HOST_ALIAS}:${GH_SLUG}.git"
fi

note "Verifying..."
if ssh -o StrictHostKeyChecking=accept-new -T "${SSH_HOST_ALIAS}" 2>&1 | grep -q 'successfully authenticated'; then
  note "SSH authentication OK."
else
  ssh -o StrictHostKeyChecking=accept-new -T "${SSH_HOST_ALIAS}" 2>&1 | sed 's/^/    /' || true
  note "If the line above does not mention this repository, re-check the deploy key."
fi
git fetch origin --quiet && note "git fetch OK."

# ----------------------------------------------------------- 4. pre-push ----
say "4. Pre-push hook"
HOOK="${REPO_ROOT}/.git/hooks/pre-push"
cat > "${HOOK}" <<'HOOKEOF'
#!/usr/bin/env bash
# Quick pressure test before anything leaves this machine.
# Blocks the push on a CRITICAL finding only. `git push --no-verify` overrides.
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)"
PY="${ROOT}/.hippo/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"
echo "pre-push: running quick pressure test..."
"${PY}" -m agents.pressure_test --quick --fail-on critical \
        --out reports/pressure_test --quiet
rc=$?
if [[ $rc -ne 0 ]]; then
  echo
  echo "pre-push: CRITICAL findings — see reports/pressure_test/report.md"
  echo "          push anyway with: git push --no-verify"
fi
exit $rc
HOOKEOF
chmod +x "${HOOK}"
note "Installed ${HOOK}"

# ---------------------------------------------------------- 5. scheduling ---
if [[ "${DO_SCHEDULE}" -eq 1 ]]; then
  say "5. Nightly run (systemd user timer)"
  UNIT_DIR="${HOME}/.config/systemd/user"
  mkdir -p "${UNIT_DIR}"

  cat > "${UNIT_DIR}/hippo-pressure-test.service" <<EOF
[Unit]
Description=hippo pipeline pressure test
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=${REPO_ROOT}
ExecStart=/usr/bin/env bash ${REPO_ROOT}/scripts/run_pressure_test.sh
EOF

  cat > "${UNIT_DIR}/hippo-pressure-test.timer" <<'EOF'
[Unit]
Description=Nightly hippo pipeline pressure test

[Timer]
OnCalendar=*-*-* 03:30:00
RandomizedDelaySec=900
Persistent=true

[Install]
WantedBy=timers.target
EOF

  systemctl --user daemon-reload
  systemctl --user enable --now hippo-pressure-test.timer
  note "Enabled. Next run: $(systemctl --user list-timers hippo-pressure-test.timer --no-pager | sed -n 2p)"
  note "Keep it running when you are logged out:  sudo loginctl enable-linger \$USER"
else
  say "5. Nightly run"
  note "Skipped. Re-run with --schedule to install the systemd timer,"
  note "or add to crontab -e:"
  note "  30 3 * * * cd ${REPO_ROOT} && bash scripts/run_pressure_test.sh >> /tmp/hippo-agent.log 2>&1"
fi

say "Done."
note "Manual run:    python -m agents.pressure_test"
note "Report:        ${REPO_ROOT}/reports/pressure_test/report.md"
