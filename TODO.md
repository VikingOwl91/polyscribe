- create folder models/ if not present -> use /usr/share/polyscribe/models/ for release version, use ./models/ for development version
- create missing folders for output files

- for merging (command line flag) -> if not present, treat each file as separate output (--merge | -m)
- for merge + separate output -> if present, treat each file as separate output and also output a merged version (--merge-and-separate)
- set speaker-names per input-file -> prompt user for each file if flag is set (--set-speaker-names)

- add support for video files -> use ffmpeg to extract audio

- detect gpus and use them
- add error handling
- add tests

- add verbose flag (--verbose | -v) + add logging

- refactor into multiple files

- rename project to "PolyScribe"

- add documentation
- package into executable
- add CI
- add package build for arch linux
