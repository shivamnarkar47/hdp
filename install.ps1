# install.ps1 - kaal installer for Windows (PowerShell 5.1+ / 7+)
# NOTE: written and reviewed on Linux; not executed here. The logic mirrors
# install.sh. Test on a real Windows machine before relying on it.
#
# Overrides (env): KAAL_REPO_URL, KAAL_INSTALL_DIR, KAAL_BIN_DIR
#Requires -Version 5.1

$ErrorActionPreference = 'Stop'
# PowerShell 7.3+: make native nonzero exits throw like cmdlet errors.
# Harmless no-op on Windows PowerShell 5.1.
$PSNativeCommandUseErrorActionPreference = $true

$RepoUrl    = if ($env:KAAL_REPO_URL)    { $env:KAAL_REPO_URL }    else { 'https://github.com/shivamnarkar47/kaal.git' }
$InstallDir = if ($env:KAAL_INSTALL_DIR) { $env:KAAL_INSTALL_DIR } else { Join-Path $HOME '.local\share\kaal' }
$BinDir     = if ($env:KAAL_BIN_DIR)     { $env:KAAL_BIN_DIR }     else { Join-Path $HOME '.local\bin' }

function Test-KaalPythonVersion {
    param([string]$VersionText)
    if ($VersionText -match 'Python\s+(\d+)\.(\d+)') {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        return ($major -gt 3) -or ($major -eq 3 -and $minor -ge 12)
    }
    return $false
}

function Assert-KaalNative {
    # Throw when the last native command failed. PS 5.1 does not raise on
    # nonzero exits, so every git/uv/pip/tar call must pass through here.
    if ($LASTEXITCODE -ne 0) {
        throw "command failed with exit code $LASTEXITCODE"
    }
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
    if (Test-KaalPythonVersion ($out | Out-String)) {
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
    Assert-KaalNative
}
elseif (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "Cloning $RepoUrl into $InstallDir"
    git clone $RepoUrl $InstallDir
    Assert-KaalNative
}
else {
    Write-Host "git not found; downloading a tarball of the main branch"
    $Tarball = Join-Path $env:TEMP 'kaal.tar.gz'
    Invoke-WebRequest -UseBasicParsing "$($RepoUrl.TrimEnd('.git'))/archive/refs/heads/main.tar.gz" -OutFile $Tarball
    # bsdtar ships with Windows 10 1803+ and reads .tar.gz directly.
    if (Get-Command tar -ErrorAction SilentlyContinue) {
        tar -xf $Tarball -C $InstallDir --strip-components=1
        Assert-KaalNative
    } else {
        # Rare: neither git nor tar. Decompress to a plain .tar with the
        # .NET GZipStream, then let Expand-Archive (ZIP-only) try it. If the
        # system still cannot read that .tar (Windows PowerShell 5.1),
        # installing Git for Windows is the fix.
        $PlainTar = Join-Path $env:TEMP 'kaal.tar'
        $src = [System.IO.File]::OpenRead($Tarball)
        $gz = [System.IO.Compression.GZipStream]::new(
            $src, [System.IO.Compression.CompressionMode]::Decompress)
        $dst = [System.IO.File]::Create($PlainTar)
        try {
            $gz.CopyTo($dst)
        } finally {
            $dst.Dispose(); $gz.Dispose(); $src.Dispose()
        }
        try {
            Expand-Archive -Path $PlainTar -DestinationPath $InstallDir -Force
        } catch {
            Write-Host "Error: could not extract the source archive. Install Git for Windows (https://git-scm.com/download/win) and re-run." -ForegroundColor Red
            exit 1
        }
    }
}

# --- Virtual environment ----------------------------------------------------
$VenvPython = Join-Path $InstallDir '.venv\Scripts\python.exe'
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "Creating virtual environment with uv"
    if (-not (Test-Path $VenvPython)) {
        uv venv (Join-Path $InstallDir '.venv')
        Assert-KaalNative
    }
    Push-Location $InstallDir
    try {
        uv pip install --python $VenvPython .
        Assert-KaalNative
    } finally {
        Pop-Location
    }
} else {
    Write-Host "Creating virtual environment with python -m venv"
    if ($PythonArg) {
        & $PythonCmd $PythonArg -m venv (Join-Path $InstallDir '.venv')
    } else {
        & $PythonCmd -m venv (Join-Path $InstallDir '.venv')
    }
    Assert-KaalNative
    & $VenvPython -m pip install .
    Assert-KaalNative
}

# --- Launcher ----------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
$Launcher = Join-Path $BinDir 'kaal.cmd'
@"
@echo off
"$VenvPython" -m harness %*
"@ | Set-Content -Path $Launcher -Encoding ASCII

# --- PATH hint ---------------------------------------------------------------
$PathEntries = @($env:PATH -split ';' | Where-Object { $_ })
if ($PathEntries -notcontains $BinDir) {
    Write-Host "NOTE: $BinDir is not on your PATH. Add it with:"
    Write-Host "  setx PATH `"$env:PATH;$BinDir`""
    Write-Host "(or via System Properties > Environment Variables)"
}

# --- Success -----------------------------------------------------------------
$RawBase = $RepoUrl -replace '^https://github\.com/', 'https://raw.githubusercontent.com/' -replace '\.git$', ''
Write-Host ""
Write-Host "kaal installed successfully."
Write-Host "  Install dir: $InstallDir"
Write-Host "  Launcher:    $Launcher"
Write-Host ""
Write-Host "Try:  kaal --help"
Write-Host "Reinstall/update:  irm $RawBase/install.ps1 | iex"
Write-Host "API key: set OPENCODE_API_KEY in your environment, or let the harness read the omp auth store."
exit 0
