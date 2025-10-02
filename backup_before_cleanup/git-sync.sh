#!/bin/bash
set -e
git fetch origin
git pull --rebase origin main
git add .
git commit -m "sync: auto update local changes with remote"
git push origin main
