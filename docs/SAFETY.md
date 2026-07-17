# Safety and Clinical Boundary

RehabEdge-Vision.AI is a developer reference project. It is not a medical device.

## Allowed output types

The default build may emit operational review events:

- person present in a zone
- no person detected
- second person present
- object detected in a zone
- camera/source problem
- possible session start

## Outputs to avoid without validation

Do not claim these without a validated clinical model and regulatory review:

- fall detected
- fall risk score
- diagnosis
- impairment severity
- clinical progress
- therapy quality
- safe to walk independently
- safe to progress exercise
- safe to remove assistive device

## Privacy stance

- Run locally by default.
- Prefer event logs over video retention.
- If saving review clips, define retention and consent rules.
- Avoid cloud transfer unless explicitly configured and reviewed.
- Treat home spatial/video data as sensitive health-context data.

## Human-in-the-loop stance

Events should feed a clinician/caregiver review process. They should not trigger unsupervised therapy changes.
