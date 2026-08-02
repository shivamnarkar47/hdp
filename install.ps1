# install.ps1 - hdp installer for Windows (PowerShell 5.1+)
# NOTE: written and reviewed on Linux; not executed here. The logic mirrors
# install.sh. Test on a real Windows machine before relying on it.
#
# Overrides (env): HDP_REPO_URL, HDP_INSTALL_DIR, HDP_BIN_DIR
$ErrorActionPreference = 'Stop'

$RepoUrl   = if ($env:HDP_REPO_URL)   { $env:HDP_REPO_URL }   else { 'https://github.com/shivamnarkar47/hdp.git' }
$InstallDir = if ($env:HDP_INSTALL_DIR) { $env:HDP_INSTALL_DIR } else { Join-Path $HOME '.local\share\hdp' }
$BinDir    = if ($env:HDP_BIN_DIR)    { $env:HDP_BIN_DIR }    else { Join-Path $HOME '.local\bin' }

function Test-HdpPythonVersion {
    param([string]$VersionText)
    $m = [regex]::Match($VersionText, 'Python\s+(\d+)\.(\d+)')
    if (-not $m.Success) { return $false }
    $major = [int]$m.Groups[1].Value
    $minor = [int]$m.Groups[2].Value
    return ($major -gt 3) -or ($major -eq 3 -and $minor -ge 12)
}

# --- Python check (>= 3.12) -------------------------------------------------
$PythonCmd = $null
$PythonArg = $null
foreach ($cand in @(@('py', '-3.12'), @('py'), @('python'))) {
    $cmd = $cand[0]
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { continue }
    if ($cand.Count -gt 1) {
        $out = & $cmd $cand[1] --version 2>&1
    } else {
        $out = & $cmd --version 2>&1
    }
    if (Test-HdpPythonVersion ($out | Out-String)) {
        $PythonCmd = $cmd
        if ($cand.Count -gt 1) { $PythonArg = $cand[1] }
        Write-Host "Found Python: $(($out | Out-String).Trim()) (via $cmd $PythonArg)"
        break
    }
}
if (-not $PythonCmd) {
    Write-Host "Error: Python 3.12 or newer is required but was not found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/ and re-run this installer."
    exit 1
}

# --- Fetch the code ---------------------------------------------------------
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
if (Test-Path (Join-Path $InstallDir '.git')) {
    Write-Host "Updating existing installation at $InstallDir"
    git -C $InstallDir pull --ff-only
}
elseif (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "Cloning $RepoUrl into $InstallDir"
    git clone $RepoUrl $InstallDir
}
else {
    Write-Host "git not found; downloading a tarball of the main branch"
    $Tarball = Join-Path $env:TEMP 'hdp.tar.gz'
    Invoke-WebRequest -UseBasicParsing "$($RepoUrl.TrimEnd('.git'))/archive/refs/heads/main.tar.gz" -OutFile $Tarball
    # bsdtar ships with Windows 10 1803+ and reads .tar.gz directly.
    if (Get-Command tar -ErrorAction SilentlyContinue) {
        tar -xf $Tarball -C $InstallDir --strip-components=1
    } else {
        # Rare: neither git nor tar. Expand-Archive only reads ZIP archives,
        # so decompress the .tar.gz to a plain .tar first. If the system's
        # Expand-Archive still cannot read that .tar (Windows PowerShell 5.1),
        # installing Git for Windows is the fix.
        $PlainTar = Join-Path $env:TEMP 'hdp.tar'
        $src = [System.IO.File]::OpenRead($Tarball)
        $gz = New-Object System.IO.Compression.GZipStream($src, [System.IO.Compression.CompressionMode]::Decompress)
        $dst = [System.IO.File]::Create($PlainTar)
        $gz.CopyTo($dst)
        $dst.Dispose(); $gz.Dispose(); $src.Dispose()
        try {
            Expand-Archive -Path $PlainTar -DestinationPath $InstallDir -Force
        } catch {
            Write-Host "Error: could not extract the source archive. Install Git for Windows (https://git-scm.com/download/win) and re-run." -ForegroundColor Red
            exit 1
        }
    }
}

# --- Virtual environment ----------------------------------------------------
Write-Host "Creating virtual environment at $InstallDir\.venv"
if ($PythonArg) {
    & $PythonCmd $PythonArg -m venv "$InstallDir\.venv"
} else {
    & $PythonCmd -m venv "$InstallDir\.venv"
}
Push-Location $InstallDir
try {
    & "$InstallDir\.venv\Scripts\python.exe" -m pip install .
} finally {
    Pop-Location
}

# --- Launcher ----------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
$Launcher = Join-Path $BinDir 'hdp.cmd'
@"
@echo off
"$InstallDir\.venv\Scripts\python.exe" -m harness %*
"@ | Set-Content -Path $Launcher -Encoding ASCII

# --- PATH hint ---------------------------------------------------------------
if ($env:PATH -notlike "*$BinDir*") {
    Write-Host "NOTE: $BinDir is not on your PATH. Add it with:"
    Write-Host "  setx PATH `"$env:PATH;$BinDir`""
    Write-Host "(or via System Properties > Environment Variables)"
}

# --- Success -----------------------------------------------------------------
$RawBase = $RepoUrl -replace '^https://github\.com/', 'https://raw.githubusercontent.com/' -replace '\.git$', ''
Write-Host ""
Write-Host "hdp installed successfully."
Write-Host "  Install dir: $InstallDir"
Write-Host "  Launcher:    $Launcher"
Write-Host ""
Write-Host "Try:  hdp --help"
Write-Host "Reinstall/update:  irm $RawBase/install.ps1 | iex"
Write-Host "API key: set OPENCODE_API_KEY in your environment, or let the harness read the omp auth store."
exit 0
