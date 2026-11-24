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

pub const CACHE_SCHEMA_VERSION: u32 = 3;
const MAX_CACHE_ENTRY_BYTES: usize = 32 * 1024 * 1024;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestKey {
    pub cache_schema_version: u32,
    pub normalization_version: u32,
    pub selection_policy_version: u32,
    pub video_id: String,
    pub requested_language: Option<String>,
    pub voxtral_executable: Option<Vec<u8>>,
    pub voxtral_executable_identity: Option<ExecutableIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cookie_file_identity: Option<ModelStat>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum SourceIdentity {
    LocalVoxtral {
        model: String,
        language: Option<String>,
        language_hint: Option<String>,
        protocol_version: u32,
    },
    LocalWhisper {
        model_name: String,
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
pub struct ExecutableIdentity {
    pub file: ModelStat,
    pub symlinks: Vec<SymlinkIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymlinkIdentity {
    pub path: Vec<u8>,
    pub target: Vec<u8>,
    pub stat: ModelStat,
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
    let metadata = std::fs::metadata(path).context("read local model metadata")?;
    if !metadata.is_file() {
        bail!("local model path is not a regular file");
    }
    Ok(model_stat_from_metadata(&metadata))
}

#[cfg(not(unix))]
pub(crate) fn model_stat(path: &Path) -> Result<ModelStat> {
    let metadata = std::fs::metadata(path).context("read local model metadata")?;
    if !metadata.is_file() {
        bail!("local model path is not a regular file");
    }
    Ok(model_stat_from_metadata(&metadata))
}

#[cfg(unix)]
fn model_stat_from_metadata(metadata: &std::fs::Metadata) -> ModelStat {
    use std::os::unix::fs::MetadataExt;

    ModelStat {
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.len(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    }
}

pub(crate) fn file_stat(file: &File) -> Result<ModelStat> {
    let metadata = file.metadata().context("inspect opened cookie file")?;
    Ok(model_stat_from_metadata(&metadata))
}

#[cfg(not(unix))]
fn model_stat_from_metadata(metadata: &std::fs::Metadata) -> ModelStat {
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default();
    ModelStat {
        device: 0,
        inode: 0,
        size: metadata.len(),
        modified_seconds: modified,
        modified_nanoseconds: 0,
        changed_seconds: 0,
        changed_nanoseconds: 0,
    }
}

#[cfg(unix)]
pub(crate) fn executable_identity(path: &Path) -> Result<ExecutableIdentity> {
    use std::collections::VecDeque;
    use std::os::unix::ffi::OsStrExt;

    enum ComponentValue {
        Parent,
        Normal(std::ffi::OsString),
    }

    fn components(path: &Path) -> VecDeque<ComponentValue> {
        path.components()
            .filter_map(|component| match component {
                Component::ParentDir => Some(ComponentValue::Parent),
                Component::Normal(name) => Some(ComponentValue::Normal(name.to_os_string())),
                Component::RootDir | Component::CurDir => None,
                Component::Prefix(_) => None,
            })
            .collect()
    }

    let absolute_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("resolve configured Voxtral executable path")?
            .join(path)
    };
    let mut pending = components(&absolute_path);
    let mut resolved = PathBuf::from("/");
    let mut symlinks = Vec::new();
    let mut link_count = 0usize;

    while let Some(component) = pending.pop_front() {
        match component {
            ComponentValue::Parent => {
                resolved.pop();
            }
            ComponentValue::Normal(name) => {
                let candidate = resolved.join(&name);
                let metadata = std::fs::symlink_metadata(&candidate).with_context(|| {
                    format!(
                        "inspect executable path {}",
                        polyscribe_core::ui::terminal_safe_path(&candidate)
                    )
                })?;
                if metadata.file_type().is_symlink() {
                    link_count += 1;
                    if link_count > 40 {
                        bail!("configured Voxtral executable exceeds the symlink limit");
                    }
                    let target = std::fs::read_link(&candidate).with_context(|| {
                        format!(
                            "read executable symlink {}",
                            polyscribe_core::ui::terminal_safe_path(&candidate)
                        )
                    })?;
                    let target_components = components(&target);
                    symlinks.push(SymlinkIdentity {
                        path: candidate.as_os_str().as_bytes().to_vec(),
                        target: target.as_os_str().as_bytes().to_vec(),
                        stat: model_stat_from_metadata(&metadata),
                    });
                    if target.is_absolute() {
                        resolved = PathBuf::from("/");
                    }
                    let mut expanded = target_components;
                    expanded.append(&mut pending);
                    pending = expanded;
                } else {
                    resolved = candidate;
                }
            }
        }
    }

    let metadata =
        std::fs::metadata(&absolute_path).context("read configured Voxtral executable metadata")?;
    if !metadata.is_file() {
        bail!("configured Voxtral executable is not a regular file");
    }
    Ok(ExecutableIdentity {
        file: model_stat_from_metadata(&metadata),
        symlinks,
    })
}

#[cfg(not(unix))]
pub(crate) fn executable_identity(_path: &Path) -> Result<ExecutableIdentity> {
    bail!("configured Voxtral executable identity requires Unix path metadata")
}

fn validate_record(record: &CacheRecord) -> Result<()> {
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
    validate_source_identity(&record.identity.source)?;
    validate_cached_cues(&record.original_cues)?;
    validate_cached_cues(&record.cues)?;
    Ok(())
}

fn validate_cached_cues(cues: &[CaptionCue]) -> Result<()> {
    let mut normalized = cues.to_vec();
    crate::youtube_transcript::normalize_and_validate_cues(&mut normalized)
        .map_err(|reason| anyhow::anyhow!("cache cues are invalid: {reason}"))?;
    if normalized != cues {
        bail!("cache cues are not normalized");
    }
    Ok(())
}

fn validate_source_identity(source: &SourceIdentity) -> Result<()> {
    let valid_text = |value: &str, max_bytes: usize| {
        !value.trim().is_empty() && value.len() <= max_bytes && !value.chars().any(char::is_control)
    };
    let valid_language = |language: &Option<String>| {
        language
            .as_deref()
            .is_none_or(crate::youtube_transcript::valid_language_code)
    };
    match source {
        SourceIdentity::LocalVoxtral {
            model,
            language,
            language_hint,
            protocol_version,
        } => {
            if *protocol_version != 1
                || !valid_text(model, 256)
                || !valid_language(language)
                || !valid_language(language_hint)
            {
                bail!("cache Voxtral identity is invalid");
            }
        }
        SourceIdentity::LocalWhisper {
            model_name,
            backend,
            language_hint,
            ..
        } => {
            if !valid_text(model_name, 4096)
                || !valid_text(backend, 32)
                || !backend
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
                || !valid_language(language_hint)
            {
                bail!("cache Whisper identity is invalid");
            }
        }
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
            Component::Normal(OsStr::new("v3")),
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

    #[test]
    fn anonymous_request_keeps_its_legacy_cache_key() {
        let legacy = r#"{"cache_schema_version":3,"normalization_version":4,"selection_policy_version":1,"video_id":"dQw4w9WgXcQ","requested_language":"de","voxtral_executable":null,"voxtral_executable_identity":null}"#;
        let request = request("dQw4w9WgXcQ");
        assert_eq!(serde_json::to_string(&request).unwrap(), legacy);
        assert_eq!(serde_json::from_str::<RequestKey>(legacy).unwrap(), request);
        assert_eq!(
            digest_json(&request).unwrap(),
            format!("{:x}", Sha256::digest(legacy.as_bytes()))
        );
    }

    fn request(video_id: &str) -> RequestKey {
        RequestKey {
            cache_schema_version: CACHE_SCHEMA_VERSION,
            normalization_version: YOUTUBE_NORMALIZATION_VERSION,
            selection_policy_version: 1,
            video_id: video_id.to_string(),
            requested_language: Some("de".to_string()),
            voxtral_executable: None,
            voxtral_executable_identity: None,
            cookie_file_identity: None,
        }
    }

    fn record(video_id: &str) -> CacheRecord {
        CacheRecord {
            schema_version: CACHE_SCHEMA_VERSION,
            normalization_version: YOUTUBE_NORMALIZATION_VERSION,
            identity: CacheIdentity {
                request: request(video_id),
                source: SourceIdentity::LocalWhisper {
                    model_name: "ggml-base.bin".to_string(),
                    model_stat: ModelStat {
                        device: 0,
                        inode: 0,
                        size: 1,
                        modified_seconds: 0,
                        modified_nanoseconds: 0,
                        changed_seconds: 0,
                        changed_nanoseconds: 0,
                    },
                    backend: "cpu".to_string(),
                    language_hint: Some("de".to_string()),
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

    fn stat(size: u64) -> ModelStat {
        ModelStat {
            device: 1,
            inode: 2,
            size,
            modified_seconds: 3,
            modified_nanoseconds: 4,
            changed_seconds: 5,
            changed_nanoseconds: 6,
        }
    }

    fn write_unvalidated_record(cache: &YoutubeCache, record: &CacheRecord) {
        let identity_hash = digest_json(&record.identity).unwrap();
        cache
            .directory
            .write_atomic(
                &format!("{identity_hash}.json"),
                &serde_json::to_vec(record).unwrap(),
            )
            .unwrap();
        let pointer = CachePointer {
            schema_version: CACHE_SCHEMA_VERSION,
            identity_hash,
        };
        cache
            .directory
            .write_atomic(
                &format!(
                    "request-{}.json",
                    digest_json(&record.identity.request).unwrap()
                ),
                &serde_json::to_vec(&pointer).unwrap(),
            )
            .unwrap();
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
    fn cache_request_distinguishes_raw_voxtral_path_and_file_stat() {
        let temp = tempfile::tempdir().unwrap();
        let cache = YoutubeCache::open_at(temp.path(), true).unwrap();
        let mut cached_request = request("voxtral-identity");
        cached_request.voxtral_executable = Some(vec![b'/', b't', b'm', b'p', b'/', 0xff]);
        cached_request.voxtral_executable_identity = Some(ExecutableIdentity {
            file: stat(10),
            symlinks: Vec::new(),
        });
        let mut cached_record = record("voxtral-identity");
        cached_record.identity.request = cached_request.clone();
        cache.store(&cached_record).unwrap();
        assert!(matches!(
            cache.lookup(&cached_request).unwrap(),
            Lookup::Hit(_)
        ));

        let mut replaced_request = cached_request.clone();
        replaced_request
            .voxtral_executable_identity
            .as_mut()
            .unwrap()
            .file = stat(11);
        assert!(matches!(
            cache.lookup(&replaced_request).unwrap(),
            Lookup::Miss
        ));

        let mut unconfigured_request = cached_request;
        unconfigured_request.voxtral_executable = None;
        unconfigured_request.voxtral_executable_identity = None;
        assert!(matches!(
            cache.lookup(&unconfigured_request).unwrap(),
            Lookup::Miss
        ));
    }

    #[cfg(unix)]
    #[test]
    fn executable_identity_captures_regular_file() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let executable = temp.path().join("voxtral");
        std::fs::write(&executable, b"binary").unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();

        let identity = executable_identity(&executable).unwrap();

        assert_eq!(identity.file, model_stat(&executable).unwrap());
        assert!(identity.symlinks.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn executable_identity_is_stable_without_changes() {
        let temp = tempfile::tempdir().unwrap();
        let executable = temp.path().join("voxtral");
        std::fs::write(&executable, b"binary").unwrap();

        assert_eq!(
            executable_identity(&executable).unwrap(),
            executable_identity(&executable).unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn executable_identity_tracks_final_symlink_retarget_and_restore() {
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::{MetadataExt, symlink};

        let temp = tempfile::tempdir().unwrap();
        let real_file = temp.path().join("voxtral");
        std::fs::write(&real_file, b"binary").unwrap();
        let target_a = temp.path().join("target-a");
        let target_b = temp.path().join("target-b");
        std::fs::hard_link(&real_file, &target_a).unwrap();
        std::fs::hard_link(&real_file, &target_b).unwrap();

        let executable = temp.path().join("configured-voxtral");
        let replacement = temp.path().join("replacement-link");
        let original_hold = temp.path().join("original-link-hold");
        let retarget_hold = temp.path().join("retarget-link-hold");
        let restore = temp.path().join("restore-link");
        symlink(&target_a, &executable).unwrap();
        symlink(&target_b, &replacement).unwrap();

        let initial = executable_identity(&executable).unwrap();
        std::fs::rename(&executable, &original_hold).unwrap();
        std::fs::rename(&replacement, &executable).unwrap();
        let retargeted = executable_identity(&executable).unwrap();
        symlink(&target_a, &restore).unwrap();
        std::fs::rename(&executable, &retarget_hold).unwrap();
        std::fs::rename(&restore, &executable).unwrap();
        let restored = executable_identity(&executable).unwrap();

        assert_eq!(initial.file, retargeted.file);
        assert_eq!(initial.file, restored.file);
        assert_eq!(initial.file, model_stat(&real_file).unwrap());
        assert_ne!(initial.symlinks, retargeted.symlinks);
        assert_ne!(initial.symlinks, restored.symlinks);
        assert_ne!(retargeted.symlinks, restored.symlinks);
        assert_eq!(initial.symlinks[0].target, target_a.as_os_str().as_bytes());
        assert_eq!(
            retargeted.symlinks[0].target,
            target_b.as_os_str().as_bytes()
        );
        assert_eq!(restored.symlinks[0].target, target_a.as_os_str().as_bytes());

        let original_inode = std::fs::symlink_metadata(&original_hold).unwrap().ino();
        let retarget_inode = std::fs::symlink_metadata(&retarget_hold).unwrap().ino();
        let restored_inode = std::fs::symlink_metadata(&executable).unwrap().ino();
        assert_ne!(original_inode, retarget_inode);
        assert_ne!(original_inode, restored_inode);
        assert_ne!(retarget_inode, restored_inode);
    }

    #[cfg(unix)]
    #[test]
    fn executable_identity_tracks_parent_symlink_with_regular_final_component() {
        use std::os::unix::fs::{MetadataExt, symlink};

        let temp = tempfile::tempdir().unwrap();
        let first_dir = temp.path().join("first-dir");
        let second_dir = temp.path().join("second-dir");
        std::fs::create_dir(&first_dir).unwrap();
        std::fs::create_dir(&second_dir).unwrap();
        let first_file = first_dir.join("voxtral");
        let second_file = second_dir.join("voxtral");
        std::fs::write(&first_file, b"binary").unwrap();
        std::fs::hard_link(&first_file, &second_file).unwrap();

        let parent_link = temp.path().join("configured-dir");
        let replacement = temp.path().join("replacement-dir-link");
        let original_hold = temp.path().join("original-dir-link-hold");
        let retarget_hold = temp.path().join("retarget-dir-link-hold");
        let restore = temp.path().join("restore-dir-link");
        symlink(&first_dir, &parent_link).unwrap();
        symlink(&second_dir, &replacement).unwrap();

        let executable = parent_link.join("voxtral");
        let initial = executable_identity(&executable).unwrap();
        assert!(std::fs::symlink_metadata(&executable).unwrap().is_file());
        std::fs::rename(&parent_link, &original_hold).unwrap();
        std::fs::rename(&replacement, &parent_link).unwrap();
        let retargeted = executable_identity(&executable).unwrap();
        symlink(&first_dir, &restore).unwrap();
        std::fs::rename(&parent_link, &retarget_hold).unwrap();
        std::fs::rename(&restore, &parent_link).unwrap();
        let restored = executable_identity(&executable).unwrap();

        assert_eq!(initial.file, retargeted.file);
        assert_eq!(initial.file, restored.file);
        assert_eq!(initial.file, model_stat(&first_file).unwrap());
        assert_eq!(initial.symlinks.len(), 1);
        assert_eq!(retargeted.symlinks.len(), 1);
        assert_eq!(restored.symlinks.len(), 1);
        assert_ne!(initial.symlinks, retargeted.symlinks);
        assert_ne!(initial.symlinks, restored.symlinks);
        assert_ne!(retargeted.symlinks, restored.symlinks);

        let original_inode = std::fs::symlink_metadata(&original_hold).unwrap().ino();
        let retarget_inode = std::fs::symlink_metadata(&retarget_hold).unwrap().ino();
        let restored_inode = std::fs::symlink_metadata(&parent_link).unwrap().ino();
        assert_ne!(original_inode, retarget_inode);
        assert_ne!(original_inode, restored_inode);
        assert_ne!(retarget_inode, restored_inode);
    }

    #[cfg(unix)]
    #[test]
    fn executable_identity_resolves_relative_nested_links_and_parent_segments() {
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let real_dir = temp.path().join("real");
        let links_dir = temp.path().join("links");
        std::fs::create_dir(&real_dir).unwrap();
        std::fs::create_dir(&links_dir).unwrap();
        let real_file = real_dir.join("voxtral");
        std::fs::write(&real_file, b"binary").unwrap();
        let configured = temp.path().join("configured-voxtral");
        let nested = links_dir.join("next");
        symlink(Path::new("links/next"), &configured).unwrap();
        symlink(Path::new("../real/voxtral"), &nested).unwrap();

        let identity = executable_identity(&configured).unwrap();

        assert_eq!(identity.file, model_stat(&real_file).unwrap());
        assert_eq!(identity.symlinks.len(), 2);
        assert_eq!(identity.symlinks[0].path, configured.as_os_str().as_bytes());
        assert_eq!(identity.symlinks[0].target, b"links/next");
        assert_eq!(identity.symlinks[1].path, nested.as_os_str().as_bytes());
        assert_eq!(identity.symlinks[1].target, b"../real/voxtral");
    }

    #[cfg(unix)]
    #[test]
    fn executable_identity_resolves_absolute_symlink_destination() {
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let real_file = temp.path().join("voxtral");
        std::fs::write(&real_file, b"binary").unwrap();
        let configured = temp.path().join("configured-voxtral");
        symlink(&real_file, &configured).unwrap();

        let identity = executable_identity(&configured).unwrap();

        assert_eq!(identity.file, model_stat(&real_file).unwrap());
        assert_eq!(identity.symlinks.len(), 1);
        assert_eq!(
            identity.symlinks[0].target,
            real_file.as_os_str().as_bytes()
        );
    }

    #[cfg(unix)]
    #[test]
    fn executable_identity_rejects_symlink_loops_and_dangling_links() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("first");
        let second = temp.path().join("second");
        symlink(&second, &first).unwrap();
        symlink(&first, &second).unwrap();
        assert!(executable_identity(&first).is_err());

        let dangling = temp.path().join("dangling");
        symlink(temp.path().join("missing"), &dangling).unwrap();
        assert!(executable_identity(&dangling).is_err());
    }

    #[test]
    fn cache_rejects_unsafe_cues_and_source_identity_on_read_and_write() {
        let temp = tempfile::tempdir().unwrap();
        let cache = YoutubeCache::open_at(temp.path(), true).unwrap();

        let mut blank_line = record("unsafe-blank-line");
        blank_line.cues[0].text = "safe\n\nforged".to_string();
        assert!(cache.store(&blank_line).is_err());
        write_unvalidated_record(&cache, &blank_line);
        assert!(matches!(
            cache.lookup(&request("unsafe-blank-line")).unwrap(),
            Lookup::Invalid
        ));

        let mut terminal_control = record("unsafe-terminal-control");
        terminal_control.original_cues[0].text = "safe\u{1b}[31m".to_string();
        assert!(cache.store(&terminal_control).is_err());
        write_unvalidated_record(&cache, &terminal_control);
        assert!(matches!(
            cache.lookup(&request("unsafe-terminal-control")).unwrap(),
            Lookup::Invalid
        ));

        let mut unsafe_speaker = record("unsafe-speaker");
        unsafe_speaker.cues[0].speaker = Some("speaker\nforged".to_string());
        assert!(cache.store(&unsafe_speaker).is_err());
        write_unvalidated_record(&cache, &unsafe_speaker);
        assert!(matches!(
            cache.lookup(&request("unsafe-speaker")).unwrap(),
            Lookup::Invalid
        ));

        let mut unsafe_model = record("unsafe-model");
        let SourceIdentity::LocalWhisper { model_name, .. } = &mut unsafe_model.identity.source
        else {
            unreachable!();
        };
        *model_name = "model\u{1b}[31m".to_string();
        assert!(cache.store(&unsafe_model).is_err());
        write_unvalidated_record(&cache, &unsafe_model);
        assert!(matches!(
            cache.lookup(&request("unsafe-model")).unwrap(),
            Lookup::Invalid
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
        let directory = temp.path().join("polyscribe/youtube/v3");
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
