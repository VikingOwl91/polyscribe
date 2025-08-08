- update the project to no more use features
- update last_model to be only used during one run

- rename project to "PolyScribe"

- update local models using hashes (--update-models)
- create folder models/ if not present -> use /usr/share/polyscribe/models/ for release version, use ./models/ for development version
- create missing folders for output files

- for merging (command line flag) -> if not present, treat each file as separate output (--merge | -m)
- for merge + separate output -> if present, treat each file as separate output and also output a merged version (--merge-and-separate)
- set speaker-names per input-file -> prompt user for each file if flag is set (--set-speaker-names)
- fix cli output for model display
- refactor into proper cli app

- add support for video files -> use ffmpeg to extract audio

- detect gpus and use them
- add error handling
- add tests

- add verbose flag (--verbose | -v) + add logging

- add documentation
- package into executable
- add CI
- add package build for arch linux
