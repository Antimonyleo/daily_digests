#!/usr/bin/env bash
# macOS: double-click this file in Finder to start DailyDigest. It runs the
# one-command installer, which sets up uv on first run and opens the app in your
# browser. (If macOS blocks it the first time: right-click -> Open, or
# System Settings -> Privacy & Security -> Open Anyway.)
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./install.sh
