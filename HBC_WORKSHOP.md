# HBC offline character workshop

Development adapter for Foundry. This branch retains the upstream workflow and adds `hbc_prepare.py`, an offline face-mesh preparation entrypoint. It preserves the source soundtrack without pitch shifting. Use your own clean dialogue recording: this route does not separate speech from music.

## Setup

Use Python 3.12 and two separate environments. For the main environment, create `.venv` and install `requirements.txt`. On Unix, run `bash setup-face-mesh.sh` for the isolated face environment. On Windows, create `.venv-face` with Python 3.12 and install the same pinned packages listed in that script using `.venv-face\Scripts\python.exe -m pip install`. The HBC adapter selects the native Windows interpreter path automatically; Windows execution has not yet been tested.

Run the main environment's Python with:

```
python hbc_prepare.py /absolute/path/to/source.mp4 /absolute/path/to/new-empty-job-folder
```

The output is a JSON result with the prepared clip, content hash and guidance fingerprint. Existing nonempty jobs are refused. Every frame must have a tracked face; incomplete tracking stops the job. Review the full output manually even when tracking succeeds. The face worker may download its model weights on first use; no source footage is submitted by this adapter.

Set Foundry's `HBC_GENJUTSU_ROOT` to this checkout's absolute path before launching the application. Foundry imports the resulting clip into its project library and saves review against its content hashes. Foundry rendering is not connected in this development slice.

## Changes and tests

`audio_workflow.py` supports zero-semitone preparation without Rubber Band. `seedance_bridge.py` respects the clean-audio manifest when building its prompt. Legacy default pitch behavior is unchanged. The upstream submission path still uses external media hosting and billable APIs; invoking it is separate from local preparation.

Run `python -m unittest discover -s tests` in the main environment. Before session reset, 15 tests passed, including a real encoded 440 Hz tone preserved without pitch change. Tracking in this test is a fixture, and provider requests are mocked. No live face-tracking quality, native Windows installation or billable generation has been validated.

Upstream MIT attribution remains in LICENSE. HBC changes do not add TEIN artwork or example footage.
