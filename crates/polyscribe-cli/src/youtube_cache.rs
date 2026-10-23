// SPDX-License-Identifier: MIT

use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use polyscribe_core::youtube::{CaptionCue, NormalizationStats, YOUTUBE_NORMALIZATION_VERSION};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const CACHE_SCHEMA_VERSION: u32 = 1;
const MAX_CACHE_ENTRY_BYTES: usize = 32 * 1024 * 1024;
const MAX_CUES: usize = 200_000;
const MAX_CUE_TEXT_BYTES: usize = 64 * 1024;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestMode {
    Default,
    CaptionsOnly,
    AudioOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestKey {
    pub cache_schema_version: u32,
    pub normalization_version: u32,
    pub selection_policy_version: u32,
    pub video_id: String,
    pub requested_language: Option<String>,
    pub prefer_manual: bool,
    pub mode: RequestMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum SourceIdentity {
    YoutubeCaptions {
        selected_language: String,
        automatic: bool,
        translated: bool,
    },
    LocalWhisper {
        model_sha256: String,
        model_stat: ModelStat,
        backend: String,
        language_hint: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStat {
    pub device: u64,
    pub inode: u64,
    pub size: u64,
    pub modified_seconds: i64,
    pub modified_nanoseconds: i64,
    pub changed_seconds: i64,
    pub changed_nanoseconds: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheIdentity {
    pub request: RequestKey,
    pub source: SourceIdentity,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VideoSummary {
    pub title: Option<String>,
    pub uploader: Option<String>,
    pub duration_seconds: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheRecord {
    pub schema_version: u32,
    pub normalization_version: u32,
    pub identity: CacheIdentity,
    pub acquired_at: String,
    pub metadata: VideoSummary,
    pub original_cues: Vec<CaptionCue>,
    pub cues: Vec<CaptionCue>,
    pub normalization: NormalizationStats,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CachePointer {
    schema_version: u32,
    identity_hash: String,
}

pub(crate) enum Lookup {
    Miss,
    Invalid,
    Hit(Box<CacheRecord>),
}

pub(crate) struct YoutubeCache {
    directory: Directory,
}

impl YoutubeCache {
    pub(crate) fn open_default(create: bool) -> Result<Option<Self>> {
        let base = cache_base_directory();
        let Some(base) = base else {
            return Ok(None);
        };
        Self::open_at(&base, create).map(Some)
    }

    pub(crate) fn open_at(base: &Path, create: bool) -> Result<Self> {
        Ok(Self {
            directory: Directory::open(base, create)?,
        })
    }

    pub(crate) fn lookup(&self, request: &RequestKey) -> Result<Lookup> {
        let request_hash = digest_json(request)?;
        let pointer_name = format!("request-{request_hash}.json");
        let Some(pointer_bytes) = self.directory.read(&pointer_name)? else {
            return Ok(Lookup::Miss);
        };
        let pointer: CachePointer = match serde_json::from_slice(&pointer_bytes) {
            Ok(pointer) => pointer,
            Err(_) => return Ok(Lookup::Invalid),
        };
        if pointer.schema_version != CACHE_SCHEMA_VERSION || !is_sha256(&pointer.identity_hash) {
            return Ok(Lookup::Invalid);
        }
        let entry_name = format!("{}.json", pointer.identity_hash);
        let Some(record_bytes) = self.directory.read(&entry_name)? else {
            return Ok(Lookup::Invalid);
        };
        let record: CacheRecord = match serde_json::from_slice(&record_bytes) {
            Ok(record) => record,
            Err(_) => return Ok(Lookup::Invalid),
        };
        if record.schema_version != CACHE_SCHEMA_VERSION
            || record.normalization_version != YOUTUBE_NORMALIZATION_VERSION
            || &record.identity.request != request
            || digest_json(&record.identity)? != pointer.identity_hash
            || validate_record(&record).is_err()
        {
            return Ok(Lookup::Invalid);
        }
        Ok(Lookup::Hit(Box::new(record)))
    }

    pub(crate) fn store(&self, record: &CacheRecord) -> Result<()> {
        validate_record(record).context("refusing to cache an invalid transcript")?;
        if record.schema_version != CACHE_SCHEMA_VERSION
            || record.normalization_version != YOUTUBE_NORMALIZATION_VERSION
        {
            bail!("refusing to cache a transcript with an unsupported schema");
        }
        let identity_hash = digest_json(&record.identity)?;
        let entry_name = format!("{identity_hash}.json");
        let entry = serde_json::to_vec(record).context("serialize YouTube transcript cache")?;
        if entry.len() > MAX_CACHE_ENTRY_BYTES {
            bail!("YouTube transcript is too large to cache");
        }
        self.directory.write_atomic(&entry_name, &entry)?;

        let request_hash = digest_json(&record.identity.request)?;
        let pointer_name = format!("request-{request_hash}.json");
        let pointer = CachePointer {
            schema_version: CACHE_SCHEMA_VERSION,
            identity_hash,
        };
        let pointer = serde_json::to_vec(&pointer).context("serialize YouTube cache index")?;
        self.directory.write_atomic(&pointer_name, &pointer)?;
        Ok(())
    }
}

pub(crate) fn cache_base_directory() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        let path = PathBuf::from(path);
        if path.is_absolute() {
            return Some(path);
        }
    }
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    home.is_absolute().then(|| home.join(".cache"))
}

#[cfg(unix)]
pub(crate) fn model_stat(path: &Path) -> Result<ModelStat> {
    use std::os::unix::fs::MetadataExt;
    let metadata = std::fs::metadata(path).context("read local model metadata")?;
    if !metadata.is_file() {
        bail!("local model path is not a regular file");
    }
    Ok(ModelStat {
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.len(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    })
}

#[cfg(not(unix))]
pub(crate) fn model_stat(path: &Path) -> Result<ModelStat> {
    let metadata = std::fs::metadata(path).context("read local model metadata")?;
    if !metadata.is_file() {
        bail!("local model path is not a regular file");
    }
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default();
    Ok(ModelStat {
        device: 0,
        inode: 0,
        size: metadata.len(),
        modified_seconds: modified,
        modified_nanoseconds: 0,
        changed_seconds: 0,
        changed_nanoseconds: 0,
    })
}

pub(crate) fn model_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).context("open local Whisper model")?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).context("read local Whisper model")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_record(record: &CacheRecord) -> Result<()> {
    if record.original_cues.len() > MAX_CUES || record.cues.len() > MAX_CUES {
        bail!("cache cue count exceeds limit");
    }
    if record
        .metadata
        .title
        .as_ref()
        .is_some_and(|s| s.len() > 16 * 1024)
        || record
            .metadata
            .uploader
            .as_ref()
            .is_some_and(|s| s.len() > 16 * 1024)
        || record
            .metadata
            .duration_seconds
            .is_some_and(|duration| !duration.is_finite() || duration < 0.0)
    {
        bail!("cache metadata is invalid");
    }
    let mut text_bytes = 0usize;
    for cue in record.original_cues.iter().chain(&record.cues) {
        if cue.end_ms < cue.start_ms
            || cue.text.len() > MAX_CUE_TEXT_BYTES
            || cue
                .speaker
                .as_ref()
                .is_some_and(|speaker| speaker.len() > 1024)
        {
            bail!("cache cue is invalid");
        }
        text_bytes = text_bytes.saturating_add(cue.text.len());
        if text_bytes > MAX_CACHE_ENTRY_BYTES / 2 {
            bail!("cache transcript text exceeds limit");
        }
    }
    if record
        .cues
        .windows(2)
        .any(|pair| pair[0].start_ms > pair[1].start_ms)
    {
        bail!("normalized cache cue timestamps are not monotonic");
    }
    if let SourceIdentity::LocalWhisper { model_sha256, .. } = &record.identity.source
        && !is_sha256(model_sha256)
    {
        bail!("cached model fingerprint is invalid");
    }
    Ok(())
}

fn digest_json(value: &impl Serialize) -> Result<String> {
    let serialized = serde_json::to_vec(value).context("serialize YouTube cache identity")?;
    let digest = Sha256::digest(serialized);
    Ok(format!("{digest:x}"))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(unix)]
struct Directory {
    fd: std::os::fd::OwnedFd,
}

#[cfg(unix)]
impl Directory {
    fn open(base: &Path, create: bool) -> Result<Self> {
        use std::ffi::CString;
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        use std::os::unix::ffi::OsStrExt;

        if !base.is_absolute()
            || base
                .components()
                .any(|component| matches!(component, Component::ParentDir | Component::CurDir))
        {
            bail!("cache base path must be absolute and normalized");
        }
        let base_count = base
            .components()
            .filter(|component| matches!(component, Component::Normal(_)))
            .count();
        let root = unsafe {
            libc::open(
                c"/".as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
            )
        };
        if root < 0 {
            return Err(io::Error::last_os_error()).context("open cache root directory");
        }
        let mut current = unsafe { OwnedFd::from_raw_fd(root) };
        let mut normal_index = 0usize;
        for component in base.components().chain([
            Component::Normal(OsStr::new("polyscribe")),
            Component::Normal(OsStr::new("youtube")),
            Component::Normal(OsStr::new("v1")),
        ]) {
            let Component::Normal(name) = component else {
                continue;
            };
            let name = CString::new(name.as_bytes()).context("cache path contains NUL")?;
            let mut next = unsafe {
                libc::openat(
                    current.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY
                        | libc::O_DIRECTORY
                        | libc::O_CLOEXEC
                        | libc::O_NOFOLLOW
                        | libc::O_NONBLOCK,
                )
            };
            if next < 0 && io::Error::last_os_error().kind() == io::ErrorKind::NotFound && create {
                if unsafe { libc::mkdirat(current.as_raw_fd(), name.as_ptr(), 0o700) } < 0 {
                    let error = io::Error::last_os_error();
                    if error.kind() != io::ErrorKind::AlreadyExists {
                        return Err(error).context("create YouTube cache directory");
                    }
                }
                next = unsafe {
                    libc::openat(
                        current.as_raw_fd(),
                        name.as_ptr(),
                        libc::O_RDONLY
                            | libc::O_DIRECTORY
                            | libc::O_CLOEXEC
                            | libc::O_NOFOLLOW
                            | libc::O_NONBLOCK,
                    )
                };
            }
            if next < 0 {
                return Err(io::Error::last_os_error())
                    .context("open YouTube cache directory without following symlinks");
            }
            let next = unsafe { OwnedFd::from_raw_fd(next) };
            let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
            if unsafe { libc::fstat(next.as_raw_fd(), stat.as_mut_ptr()) } < 0 {
                return Err(io::Error::last_os_error()).context("inspect YouTube cache directory");
            }
            let stat = unsafe { stat.assume_init() };
            let effective_uid = unsafe { libc::geteuid() };
            let is_cache_directory = normal_index > base_count;
            let is_shared_parent = normal_index == base_count;
            let is_base_directory = normal_index + 1 == base_count;
            let accepted_parent = (stat.st_uid == effective_uid || stat.st_uid == 0)
                && (stat.st_mode & 0o022 == 0
                    || (stat.st_uid == 0 && stat.st_mode & libc::S_ISVTX != 0));
            let accepted_base = stat.st_uid == effective_uid && stat.st_mode & 0o022 == 0;
            let accepted_cache = stat.st_uid == effective_uid && stat.st_mode & 0o077 == 0;
            let accepted = if is_cache_directory {
                accepted_cache
            } else if is_shared_parent || is_base_directory {
                accepted_base
            } else {
                accepted_parent
            };
            if !accepted {
                bail!("YouTube cache directory ownership or permissions are unsafe");
            }
            current = next;
            normal_index += 1;
        }
        Ok(Self { fd: current })
    }

    fn read(&self, name: &str) -> Result<Option<Vec<u8>>> {
        use std::ffi::CString;
        use std::os::fd::{AsRawFd, FromRawFd};

        let name = CString::new(name).context("invalid cache filename")?;
        let fd = unsafe {
            libc::openat(
                self.fd.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
            )
        };
        if fd < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::NotFound {
                return Ok(None);
            }
            return Err(error).context("open YouTube cache entry without following symlinks");
        }
        let owned = unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) };
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        if unsafe { libc::fstat(owned.as_raw_fd(), stat.as_mut_ptr()) } < 0 {
            return Err(io::Error::last_os_error()).context("inspect YouTube cache entry");
        }
        let stat = unsafe { stat.assume_init() };
        if stat.st_mode & libc::S_IFMT != libc::S_IFREG
            || stat.st_uid != unsafe { libc::geteuid() }
            || stat.st_mode & 0o077 != 0
            || stat.st_nlink != 1
            || stat.st_size < 0
            || stat.st_size as usize > MAX_CACHE_ENTRY_BYTES
        {
            bail!("YouTube cache entry type, owner, mode, links, or size are unsafe");
        }
        let file = File::from(owned);
        let mut bytes = Vec::with_capacity(stat.st_size as usize);
        file.take((MAX_CACHE_ENTRY_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .context("read YouTube cache entry")?;
        if bytes.len() > MAX_CACHE_ENTRY_BYTES {
            bail!("YouTube cache entry exceeds the size limit");
        }
        Ok(Some(bytes))
    }

    fn write_atomic(&self, name: &str, bytes: &[u8]) -> Result<()> {
        use std::ffi::CString;
        use std::os::fd::{AsRawFd, FromRawFd};

        if bytes.len() > MAX_CACHE_ENTRY_BYTES {
            bail!("YouTube cache entry exceeds the size limit");
        }
        let destination = CString::new(name).context("invalid cache filename")?;
        for _ in 0..16 {
            let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let temporary = CString::new(format!(".tmp-{}-{counter}", std::process::id()))
                .expect("temporary cache filename has no NUL");
            let fd = unsafe {
                libc::openat(
                    self.fd.as_raw_fd(),
                    temporary.as_ptr(),
                    libc::O_WRONLY
                        | libc::O_CREAT
                        | libc::O_EXCL
                        | libc::O_CLOEXEC
                        | libc::O_NOFOLLOW,
                    0o600,
                )
            };
            if fd < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::AlreadyExists {
                    continue;
                }
                return Err(error).context("create temporary YouTube cache entry");
            }
            let file = unsafe { File::from_raw_fd(fd) };
            let mut temporary_guard = TemporaryFile {
                directory_fd: self.fd.as_raw_fd(),
                name: temporary.clone(),
                committed: false,
            };
            let mut file = file;
            file.write_all(bytes)
                .context("write temporary YouTube cache entry")?;
            file.sync_all()
                .context("sync temporary YouTube cache entry")?;
            drop(file);
            if unsafe {
                libc::renameat(
                    self.fd.as_raw_fd(),
                    temporary.as_ptr(),
                    self.fd.as_raw_fd(),
                    destination.as_ptr(),
                )
            } < 0
            {
                return Err(io::Error::last_os_error())
                    .context("atomically replace YouTube cache entry");
            }
            temporary_guard.committed = true;
            if unsafe { libc::fsync(self.fd.as_raw_fd()) } < 0 {
                return Err(io::Error::last_os_error()).context("sync YouTube cache directory");
            }
            return Ok(());
        }
        bail!("could not allocate a unique temporary cache filename")
    }
}

#[cfg(unix)]
struct TemporaryFile {
    directory_fd: std::os::fd::RawFd,
    name: std::ffi::CString,
    committed: bool,
}

#[cfg(unix)]
impl Drop for TemporaryFile {
    fn drop(&mut self) {
        if !self.committed {
            unsafe {
                libc::unlinkat(self.directory_fd, self.name.as_ptr(), 0);
            }
        }
    }
}

#[cfg(not(unix))]
struct Directory;

#[cfg(not(unix))]
impl Directory {
    fn open(_base: &Path, _create: bool) -> Result<Self> {
        bail!("YouTube transcript caching requires Unix descriptor-relative file operations")
    }

    fn read(&self, _name: &str) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }

    fn write_atomic(&self, _name: &str, _bytes: &[u8]) -> Result<()> {
        bail!("YouTube transcript caching requires Unix descriptor-relative file operations")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polyscribe_core::youtube::CaptionCue;

    fn request(video_id: &str) -> RequestKey {
        RequestKey {
            cache_schema_version: CACHE_SCHEMA_VERSION,
            normalization_version: YOUTUBE_NORMALIZATION_VERSION,
            selection_policy_version: 1,
            video_id: video_id.to_string(),
            requested_language: Some("de".to_string()),
            prefer_manual: true,
            mode: RequestMode::Default,
        }
    }

    fn record(video_id: &str) -> CacheRecord {
        CacheRecord {
            schema_version: CACHE_SCHEMA_VERSION,
            normalization_version: YOUTUBE_NORMALIZATION_VERSION,
            identity: CacheIdentity {
                request: request(video_id),
                source: SourceIdentity::YoutubeCaptions {
                    selected_language: "de-DE".to_string(),
                    automatic: false,
                    translated: false,
                },
            },
            acquired_at: "2026-10-23T00:00:00Z".to_string(),
            metadata: VideoSummary::default(),
            original_cues: vec![CaptionCue {
                start_ms: 1000,
                end_ms: 2000,
                text: "Hallo".to_string(),
                speaker: None,
                speaker_marker: false,
            }],
            cues: vec![CaptionCue {
                start_ms: 1000,
                end_ms: 2000,
                text: "Hallo".to_string(),
                speaker: None,
                speaker_marker: false,
            }],
            normalization: NormalizationStats::default(),
        }
    }

    #[test]
    fn cache_roundtrips_and_separates_video_identity() {
        let temp = tempfile::tempdir().unwrap();
        let cache = YoutubeCache::open_at(temp.path(), true).unwrap();
        cache.store(&record("video-one")).unwrap();
        assert!(matches!(
            cache.lookup(&request("video-one")).unwrap(),
            Lookup::Hit(_)
        ));
        assert!(matches!(
            cache.lookup(&request("video-two")).unwrap(),
            Lookup::Miss
        ));
    }

    #[test]
    fn corrupt_and_symlinked_entries_are_never_trusted() {
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let cache = YoutubeCache::open_at(temp.path(), true).unwrap();
        let request = request("video-corrupt");
        cache.store(&record("video-corrupt")).unwrap();
        let request_hash = digest_json(&request).unwrap();
        let pointer_name = format!("request-{request_hash}.json");
        let entry_name = format!(
            "{}.json",
            digest_json(&record("video-corrupt").identity).unwrap()
        );
        let directory = temp.path().join("polyscribe/youtube/v1");
        std::fs::write(directory.join(&pointer_name), "not json").unwrap();
        assert!(matches!(cache.lookup(&request).unwrap(), Lookup::Invalid));
        std::fs::remove_file(directory.join(&pointer_name)).unwrap();
        std::fs::remove_file(directory.join(&entry_name)).unwrap();
        let outside = temp.path().join("outside.json");
        std::fs::write(
            &outside,
            serde_json::to_vec(&record("video-corrupt")).unwrap(),
        )
        .unwrap();
        std::fs::set_permissions(&outside, std::fs::Permissions::from_mode(0o600)).unwrap();
        symlink(&outside, directory.join(&entry_name)).unwrap();
        let pointer = CachePointer {
            schema_version: CACHE_SCHEMA_VERSION,
            identity_hash: entry_name.trim_end_matches(".json").to_string(),
        };
        std::fs::write(
            directory.join(&pointer_name),
            serde_json::to_vec(&pointer).unwrap(),
        )
        .unwrap();
        std::fs::set_permissions(
            directory.join(&pointer_name),
            std::fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        assert!(cache.lookup(&request).is_err());
        assert!(outside.exists());
    }

    #[test]
    fn cache_rejects_nonmonotonic_normalized_cues() {
        let mut value = record("video-time");
        value.cues.push(CaptionCue {
            start_ms: 500,
            end_ms: 1500,
            text: "Earlier".to_string(),
            speaker: None,
            speaker_marker: false,
        });
        assert!(validate_record(&value).is_err());
    }

    #[test]
    fn concurrent_refreshes_leave_a_complete_cache_record() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().to_path_buf();
        let mut writers = Vec::new();
        for _ in 0..8 {
            let base = base.clone();
            writers.push(std::thread::spawn(move || {
                let cache = YoutubeCache::open_at(&base, true).unwrap();
                cache.store(&record("video-race")).unwrap();
            }));
        }
        for writer in writers {
            writer.join().unwrap();
        }
        let cache = YoutubeCache::open_at(&base, false).unwrap();
        assert!(matches!(
            cache.lookup(&request("video-race")).unwrap(),
            Lookup::Hit(_)
        ));
    }
}
