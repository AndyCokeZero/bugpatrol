# Lark icon assets

Two distinct icons — do not mix them up.

## `bugpatrol-icon.png` — Lark **app** avatar

Paw-Patrol-style gold-rimmed red shield with a white beetle, on white.
Applied to app `cli_aac97d050d385ee9` (BugPatrol).

Regenerate from the 1024 source (repo asset == live icon):

```
python3 assets/lark-icon/fit_circle_safe.py \
  assets/lark-icon/bugpatrol-icon-source.png assets/lark-icon/bugpatrol-icon.png \
  --height-ratio 0.80 --preview /tmp/preview.png
```

The artwork sits on white, so a circular avatar crop would clip it if sized
naively — hence `fit_circle_safe.py`. It measures the farthest painted pixel
from center (the true shape radius), not the bounding-box corners. Current
setting 0.80 leaves a 22px margin; the safe maximum for this artwork is 0.859.

Apply with `lark-console`, then publish a version (icon changes need one):

```
node scripts/console_api.mjs app set-icon cli_aac97d050d385ee9 --icon assets/lark-icon/bugpatrol-icon.png
node scripts/console_api.mjs version publish cli_aac97d050d385ee9 --version <ver> --notes <notes>
```

## `group-icon.png` — Lark **bug-report group** avatar

Amber beetle under a patrol searchlight cone on full-bleed midnight blue.
Applied to every BugPatrol bug-report topic group (fived's `fived · main` and
its per-branch `fived · feature-*` children, plus the sandbox groups).

It is full-bleed, so the circular crop only removes background corners — no
circle-safe fitting needed. `group-icon.png` is a plain 512 LANCZOS resize of
`group-icon-source.png`.

New `feature-*` groups inherit this automatically: fived's
`.github/workflows/bugpatrol-feature-topic.yml` re-uploads the **current**
avatar of the `fived · main` group rather than hardcoding an `image_key`. So
changing the icon means updating `fived · main` plus a one-off sweep of the
existing groups:

```
# bot identity, needs im:resource + im:chat:update
POST /open-apis/im/v1/images   -F image_type=avatar -F image=@group-icon.png
PUT  /open-apis/im/v1/chats/{chat_id}   {"avatar": "<image_key>"}
```

One `image_key` can be reused across every chat. `PUT chats` works on
`chat_mode=topic` groups despite the docs listing `group` only.
