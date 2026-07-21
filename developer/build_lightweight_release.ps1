<#
.SYNOPSIS
Builds the self-contained Windows lightweight HABIT source release.

.DESCRIPTION
The script builds a native CPython 3.10 wheel from an isolated temporary source
copy, stages only explicitly approved repository content, verifies all vendored
binary hashes, writes a payload manifest, performs factory checks, and creates a
ZIP archive whose single top-level directory is HABIT-light-v<version>.

The release manifest intentionally excludes itself. Including a cryptographic
hash of the manifest inside that same manifest is not mathematically stable.
#>
[CmdletBinding()]
param(
    [Parameter()]
    [string]$OutputDirectory,

    [Parameter()]
    [string]$BuildPython = 'E:\conda\mconda\envs\py310\python.exe',

    [Parameter()]
    [switch]$SkipGitCleanCheck
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$script:RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$script:Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$script:ExcludedDirectoryNames = @(
    '.git',
    '.mamba',
    '.cache',
    '.mypy_cache',
    '.pytest_cache',
    '.ruff_cache',
    '__pycache__',
    'build',
    'demo_data',
    'dist',
    'tests'
)
$script:ExcludedFileExtensions = @('.pyc', '.pyo', '.pyd', '.so')

function Resolve-ExistingPath {
    <#
    .SYNOPSIS
    Resolves an existing file-system path to an absolute provider path.

    .PARAMETER Path
    The input path, which may be absolute or relative to the repository root.

    .PARAMETER ExpectedType
    The required path type: Leaf for a file or Container for a directory.

    .OUTPUTS
    System.String
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [ValidateSet('Leaf', 'Container')]
        [string]$ExpectedType
    )

    $candidate = $Path
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $script:RepoRoot $candidate
    }

    if (-not (Test-Path -LiteralPath $candidate -PathType $ExpectedType)) {
        throw "Required $ExpectedType path does not exist: $candidate"
    }

    return (Resolve-Path -LiteralPath $candidate).ProviderPath
}

function Get-RelativePosixPath {
    <#
    .SYNOPSIS
    Produces a deterministic POSIX-style path relative to a package root.

    .OUTPUTS
    System.String
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,

        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $rootPrefix = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/') +
        [System.IO.Path]::DirectorySeparatorChar
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith(
            $rootPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
        throw "Path is outside the expected root: $fullPath"
    }

    return $fullPath.Substring($rootPrefix.Length).Replace('\', '/')
}

function Test-IsExcludedDirectoryName {
    <#
    .SYNOPSIS
    Checks whether a directory name is forbidden in source and release trees.

    .OUTPUTS
    System.Boolean
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    return $script:ExcludedDirectoryNames -contains $Name.ToLowerInvariant()
}

function Copy-FilteredTree {
    <#
    .SYNOPSIS
    Recursively copies a source tree while rejecting links and build residue.

    .DESCRIPTION
    Reparse points are rejected instead of followed so a source checkout cannot
    silently pull files from outside the repository. Generated native modules
    are excluded so the wheel must compile its C extension during this build.

    .OUTPUTS
    System.Void
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,

        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Source directory does not exist: $Source"
    }

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $Source -Force) {
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse points are not allowed in release inputs: $($item.FullName)"
        }

        $target = Join-Path $Destination $item.Name
        if ($item.PSIsContainer) {
            if (Test-IsExcludedDirectoryName -Name $item.Name) {
                continue
            }
            Copy-FilteredTree -Source $item.FullName -Destination $target
            continue
        }

        if ($script:ExcludedFileExtensions -contains $item.Extension.ToLowerInvariant()) {
            continue
        }
        Copy-Item -LiteralPath $item.FullName -Destination $target
    }
}

function Copy-ExactTree {
    <#
    .SYNOPSIS
    Copies an external license tree exactly while rejecting reparse points.

    .OUTPUTS
    System.Void
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,

        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $Source -Force) {
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse points are not allowed in license inputs: $($item.FullName)"
        }

        $target = Join-Path $Destination $item.Name
        if ($item.PSIsContainer) {
            Copy-ExactTree -Source $item.FullName -Destination $target
        }
        else {
            Copy-Item -LiteralPath $item.FullName -Destination $target
        }
    }
}

function Test-IdentityMatch {
    <#
    .SYNOPSIS
    Tests whether text identifies one of an asset's accepted names.

    .OUTPUTS
    System.Boolean
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [AllowNull()]
        [object]$Value,

        [Parameter(Mandatory = $true)]
        [string[]]$Names
    )

    if ($null -eq $Value -or -not ($Value -is [string])) {
        return $false
    }

    $text = ([string]$Value).Replace('\', '/').ToLowerInvariant()
    foreach ($name in $Names) {
        if ($text.Contains($name.ToLowerInvariant())) {
            return $true
        }
    }
    return $false
}

function Find-AssetHashes {
    <#
    .SYNOPSIS
    Recursively finds SHA256 values associated with a named fixed asset.

    .DESCRIPTION
    The reader accepts either an assets array or a name-keyed JSON object. This
    keeps the build script tolerant of harmless manifest layout changes while
    still requiring exactly one unambiguous SHA256 value for every fixed asset.

    .OUTPUTS
    System.String[]
    #>
    [CmdletBinding()]
    [OutputType([string[]])]
    param(
        [AllowNull()]
        [object]$Node,

        [Parameter(Mandatory = $true)]
        [string[]]$Names,

        [Parameter()]
        [string]$Context = ''
    )

    $results = New-Object 'System.Collections.Generic.List[string]'
    if ($null -eq $Node) {
        return @()
    }

    if ($Node -is [string]) {
        $text = [string]$Node
        if (
            $text -match '^[A-Fa-f0-9]{64}$' -and
            (Test-IdentityMatch -Value $Context -Names $Names)
        ) {
            $results.Add($text.ToLowerInvariant())
        }
        return $results.ToArray()
    }

    if (
        $Node -is [System.Collections.IEnumerable] -and
        -not ($Node -is [System.Management.Automation.PSCustomObject]) -and
        -not ($Node -is [System.Collections.IDictionary])
    ) {
        $index = 0
        foreach ($child in $Node) {
            foreach ($hash in Find-AssetHashes `
                -Node $child `
                -Names $Names `
                -Context "$Context/$index") {
                $results.Add($hash)
            }
            $index++
        }
        return $results.ToArray()
    }

    $properties = @($Node.PSObject.Properties)
    $objectIdentifiesAsset = Test-IdentityMatch -Value $Context -Names $Names
    foreach ($property in $properties) {
        if (Test-IdentityMatch -Value $property.Name -Names $Names) {
            $objectIdentifiesAsset = $true
        }
        if (Test-IdentityMatch -Value $property.Value -Names $Names) {
            $objectIdentifiesAsset = $true
        }
    }

    if ($objectIdentifiesAsset) {
        foreach ($property in $properties) {
            if (
                $property.Name -match '^(sha256|sha_256|hash)$' -and
                $property.Value -is [string] -and
                ([string]$property.Value) -match '^[A-Fa-f0-9]{64}$'
            ) {
                $results.Add(([string]$property.Value).ToLowerInvariant())
            }
        }
    }

    foreach ($property in $properties) {
        $childContext = "$Context/$($property.Name)"
        foreach ($hash in Find-AssetHashes `
            -Node $property.Value `
            -Names $Names `
            -Context $childContext) {
            $results.Add($hash)
        }
    }
    return $results.ToArray()
}

function Get-ExpectedAssetHash {
    <#
    .SYNOPSIS
    Resolves one asset's unique expected hash from vendor_assets.json.

    .OUTPUTS
    System.String
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [object]$Manifest,

        [Parameter(Mandatory = $true)]
        [string[]]$LookupNames,

        [Parameter(Mandatory = $true)]
        [string]$DisplayName
    )

    $hashes = @(
        Find-AssetHashes -Node $Manifest -Names $LookupNames |
            Sort-Object -Unique
    )
    if ($hashes.Count -ne 1) {
        throw (
            "vendor_assets.json must define exactly one SHA256 for '{0}'; " +
            "found {1}." -f $DisplayName, $hashes.Count
        )
    }
    return $hashes[0]
}

function Assert-FileHash {
    <#
    .SYNOPSIS
    Verifies a file against an expected SHA256 digest.

    .OUTPUTS
    System.Void
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedHash,

        [Parameter(Mandatory = $true)]
        [string]$DisplayName
    )

    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedHash.ToLowerInvariant()) {
        throw (
            "SHA256 mismatch for '{0}'. Expected {1}, actual {2}." -f
            $DisplayName,
            $ExpectedHash,
            $actual
        )
    }
}

function Assert-BuildPython {
    <#
    .SYNOPSIS
    Requires native 64-bit CPython 3.10 on Windows for the wheel build.

    .OUTPUTS
    System.Void
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonPath
    )

    # Keep the -c payload on one line. Windows PowerShell 5.1 can otherwise
    # split a multiline here-string into multiple native argv entries.
    $probe = "import platform,struct,sys; assert platform.python_implementation()=='CPython'; assert sys.version_info[:2]==(3,10); assert sys.platform=='win32'; assert struct.calcsize('P')*8==64; print(sys.executable)"
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = & $PythonPath -I -s -c $probe 2>&1
        $probeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($probeExitCode -ne 0) {
        throw "BuildPython must be 64-bit Windows CPython 3.10: $output"
    }
}

function Assert-WheelContents {
    <#
    .SYNOPSIS
    Verifies wheel tags, package data, native extension, and pollution markers.

    .OUTPUTS
    System.Void
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$WheelPath
    )

    if (
        [System.IO.Path]::GetFileName($WheelPath) -notmatch
        '(?i)^habit-.+-cp310-cp310-win_amd64\.whl$'
    ) {
        throw "Unexpected HABIT wheel platform tag: $WheelPath"
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($WheelPath)
    try {
        $names = @($archive.Entries | ForEach-Object { $_.FullName.Replace('\', '/') })
        if (-not ($names -contains 'habit/py.typed')) {
            throw 'Built wheel is missing habit/py.typed.'
        }
        if (-not ($names | Where-Object {
                    $_ -match '(?i)^habit/.*/_sv_cmatrices[^/]*\.pyd$'
                })) {
            throw 'Built wheel is missing the native _sv_cmatrices C extension.'
        }
        if (-not ($names | Where-Object {
                    $_ -match '(?i)^habit/resources/radiomics/[^/]+\.ya?ml$'
                })) {
            throw 'Built wheel is missing bundled radiomics presets.'
        }

        $pollution = @($names | Where-Object {
                $_ -match '(?i)(^|/)(direct_url\.json|.*\.egg-link|.*\.pth)$' -or
                $_ -match '(?i)(^|/)(\.git|\.mamba|tests?|demo_data)(/|$)'
            })
        if ($pollution.Count -gt 0) {
            throw "Built wheel contains editable or repository pollution: $($pollution -join ', ')"
        }
    }
    finally {
        $archive.Dispose()
    }
}

function Write-ReleaseManifest {
    <#
    .SYNOPSIS
    Writes deterministic metadata for every payload file except the manifest.

    .OUTPUTS
    System.Void
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$PackageRoot,

        [Parameter(Mandatory = $true)]
        [string]$Version
    )

    $manifestPath = Join-Path $PackageRoot 'release_manifest.json'
    $records = @(
        Get-ChildItem -LiteralPath $PackageRoot -File -Recurse -Force |
            Where-Object { $_.FullName -ne $manifestPath } |
            ForEach-Object {
                [pscustomobject][ordered]@{
                    path   = Get-RelativePosixPath -Root $PackageRoot -Path $_.FullName
                    size   = [int64]$_.Length
                    sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).
                        Hash.ToLowerInvariant()
                }
            } |
            Sort-Object -Property path
    )

    $manifest = [ordered]@{
        schema_version    = 1
        product           = 'HABIT-light'
        version           = $Version
        hash_algorithm    = 'SHA256'
        manifest_excludes = @('release_manifest.json')
        files             = $records
    }
    $json = $manifest | ConvertTo-Json -Depth 6
    [System.IO.File]::WriteAllText($manifestPath, $json + "`r`n", $script:Utf8NoBom)
}

function Assert-ReleaseManifest {
    <#
    .SYNOPSIS
    Recomputes every payload record and rejects missing or extra manifest rows.

    .OUTPUTS
    System.Void
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$PackageRoot
    )

    $manifestPath = Join-Path $PackageRoot 'release_manifest.json'
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 |
        ConvertFrom-Json
    $payloadFiles = @(
        Get-ChildItem -LiteralPath $PackageRoot -File -Recurse -Force |
            Where-Object { $_.FullName -ne $manifestPath }
    )
    $records = @($manifest.files)
    if ($records.Count -ne $payloadFiles.Count) {
        throw (
            "release_manifest.json has {0} rows for {1} payload files." -f
            $records.Count,
            $payloadFiles.Count
        )
    }

    $recordMap = @{}
    foreach ($record in $records) {
        $path = [string]$record.path
        if ($recordMap.ContainsKey($path)) {
            throw "Duplicate release manifest path: $path"
        }
        $recordMap[$path] = $record
    }

    foreach ($file in $payloadFiles) {
        $relativePath = Get-RelativePosixPath -Root $PackageRoot -Path $file.FullName
        if (-not $recordMap.ContainsKey($relativePath)) {
            throw "File is missing from release_manifest.json: $relativePath"
        }
        $record = $recordMap[$relativePath]
        if ([int64]$record.size -ne [int64]$file.Length) {
            throw "Size mismatch in release_manifest.json: $relativePath"
        }
        Assert-FileHash `
            -Path $file.FullName `
            -ExpectedHash ([string]$record.sha256) `
            -DisplayName $relativePath
    }
}

function Assert-FactoryChecks {
    <#
    .SYNOPSIS
    Runs the complete staging-tree factory acceptance checks.

    .OUTPUTS
    System.Void
    #>
    [CmdletBinding()]
    [OutputType([void])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$PackageRoot,

        [Parameter(Mandatory = $true)]
        [object[]]$FixedAssets,

        [Parameter(Mandatory = $true)]
        [string[]]$LauncherBatEntries
    )

    $requiredEntries = @(
        'habit/_version.py',
        'habit/py.typed',
        'config',
        'installer/vendor_assets.json',
        'LICENSE',
        'README.md',
        'pyproject.toml',
        'setup.py',
        'MANIFEST.in',
        'tools/bin/dcm2niix.exe',
        'tools/bin/elastix.exe',
        'tools/bin/transformix.exe',
        'tools/micromamba/micromamba.exe',
        'licenses/micromamba',
        'release_manifest.json'
    ) + $LauncherBatEntries
    foreach ($relativePath in $requiredEntries) {
        $nativePath = Join-Path $PackageRoot ($relativePath.Replace('/', '\'))
        if (-not (Test-Path -LiteralPath $nativePath)) {
            throw "Required release entry is missing: $relativePath"
        }
    }

    $forbiddenDirectories = @(
        Get-ChildItem -LiteralPath $PackageRoot -Directory -Recurse -Force |
            Where-Object { Test-IsExcludedDirectoryName -Name $_.Name }
    )
    if ($forbiddenDirectories.Count -gt 0) {
        throw (
            'Forbidden directories found in release: ' +
            (($forbiddenDirectories | ForEach-Object {
                        Get-RelativePosixPath -Root $PackageRoot -Path $_.FullName
                    }) -join ', ')
        )
    }

    $wheelDirectory = Join-Path $PackageRoot 'tools\wheels'
    $habitWheels = @(Get-ChildItem -LiteralPath $wheelDirectory -File -Filter 'HABIT-*.whl')
    if ($habitWheels.Count -ne 1) {
        throw "Release must contain exactly one HABIT wheel; found $($habitWheels.Count)."
    }
    Assert-WheelContents -WheelPath $habitWheels[0].FullName

    foreach ($asset in $FixedAssets) {
        $stagedPath = Join-Path $PackageRoot ([string]$asset.Destination)
        Assert-FileHash `
            -Path $stagedPath `
            -ExpectedHash ([string]$asset.ExpectedHash) `
            -DisplayName ([string]$asset.Name)
    }

    $micromamba = Join-Path $PackageRoot 'tools\micromamba\micromamba.exe'
    $micromambaVersion = & $micromamba --version 2>&1
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(
            [string]($micromambaVersion -join ' ')
        )) {
        throw "Staged micromamba --version failed: $micromambaVersion"
    }

    Assert-ReleaseManifest -PackageRoot $PackageRoot
}

$lockStream = $null
$lockPath = $null
$temporaryRoot = $null
try {
    Write-Host '[HABIT] Validating repository and build inputs...'

    if (-not $SkipGitCleanCheck) {
        $gitStatus = @(
            & git -C $script:RepoRoot status --porcelain=v1 --untracked-files=all 2>&1
        )
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to inspect git status: $($gitStatus -join [Environment]::NewLine)"
        }
        if ($gitStatus.Count -gt 0) {
            throw (
                "Repository is not clean. Commit/stash changes or explicitly use " +
                "-SkipGitCleanCheck:`r`n$($gitStatus -join [Environment]::NewLine)"
            )
        }
    }
    else {
        Write-Warning 'Git cleanliness check was explicitly skipped.'
    }

    $versionFile = Join-Path $script:RepoRoot 'habit\_version.py'
    $versionText = Get-Content -LiteralPath $versionFile -Raw -Encoding UTF8
    $versionMatch = [regex]::Match(
        $versionText,
        '(?m)^\s*__version__\s*=\s*["''](?<version>[^"'']+)["'']\s*$'
    )
    if (-not $versionMatch.Success) {
        throw "Unable to read __version__ from $versionFile"
    }
    $version = $versionMatch.Groups['version'].Value
    if ($version -notmatch '^[0-9A-Za-z][0-9A-Za-z.+-]*$') {
        throw "HABIT version is unsafe for a release directory name: $version"
    }

    $vendorManifestPath = Join-Path $script:RepoRoot 'installer\vendor_assets.json'
    if (-not (Test-Path -LiteralPath $vendorManifestPath -PathType Leaf)) {
        throw "Required vendor asset manifest does not exist: $vendorManifestPath"
    }
    $vendorManifest = Get-Content -LiteralPath $vendorManifestPath -Raw -Encoding UTF8 |
        ConvertFrom-Json

    $resolvedPython = Resolve-ExistingPath -Path $BuildPython -ExpectedType Leaf
    $micromambaVendorDirectory = Join-Path `
        $script:RepoRoot `
        'installer\vendor\micromamba'
    $resolvedMicromamba = Resolve-ExistingPath `
        -Path (Join-Path $micromambaVendorDirectory 'micromamba.exe') `
        -ExpectedType Leaf
    $resolvedMicromambaLicenses = Resolve-ExistingPath `
        -Path (Join-Path $micromambaVendorDirectory 'licenses') `
        -ExpectedType Container
    Assert-BuildPython -PythonPath $resolvedPython

    $vendorSourceDirectory = Join-Path $script:RepoRoot 'installer\vendor'
    $pyradiomicsWheels = @(
        Get-ChildItem `
            -LiteralPath $vendorSourceDirectory `
            -File `
            -Filter 'pyradiomics-*-cp310-cp310-win_amd64.whl'
    )
    if ($pyradiomicsWheels.Count -ne 1) {
        throw (
            'installer\vendor must contain exactly one CPython 3.10 win_amd64 ' +
            "PyRadiomics wheel; found $($pyradiomicsWheels.Count)."
        )
    }

    $fixedAssets = @(
        [pscustomobject]@{
            Name        = $pyradiomicsWheels[0].Name
            Source      = $pyradiomicsWheels[0].FullName
            Destination = "tools\vendor\$($pyradiomicsWheels[0].Name)"
            LookupNames = @(
                $pyradiomicsWheels[0].Name,
                'pyradiomics'
            )
            ExpectedHash = $null
        },
        [pscustomobject]@{
            Name        = 'dcm2niix.exe'
            Source      = Join-Path $script:RepoRoot 'tools\bin\dcm2niix.exe'
            Destination = 'tools\bin\dcm2niix.exe'
            LookupNames = @('dcm2niix.exe', 'dcm2niix')
            ExpectedHash = $null
        },
        [pscustomobject]@{
            Name        = 'elastix.exe'
            Source      = Join-Path $script:RepoRoot 'tools\bin\elastix.exe'
            Destination = 'tools\bin\elastix.exe'
            LookupNames = @('elastix.exe', 'elastix')
            ExpectedHash = $null
        },
        [pscustomobject]@{
            Name        = 'transformix.exe'
            Source      = Join-Path $script:RepoRoot 'tools\bin\transformix.exe'
            Destination = 'tools\bin\transformix.exe'
            LookupNames = @('transformix.exe', 'transformix')
            ExpectedHash = $null
        },
        [pscustomobject]@{
            Name        = 'micromamba.exe'
            Source      = $resolvedMicromamba
            Destination = 'tools\micromamba\micromamba.exe'
            LookupNames = @('micromamba.exe', 'micromamba')
            ExpectedHash = $null
        }
    )
    foreach ($asset in $fixedAssets) {
        if (-not (Test-Path -LiteralPath $asset.Source -PathType Leaf)) {
            throw "Required fixed asset does not exist: $($asset.Source)"
        }
        $asset.ExpectedHash = Get-ExpectedAssetHash `
            -Manifest $vendorManifest `
            -LookupNames $asset.LookupNames `
            -DisplayName $asset.Name
        Assert-FileHash `
            -Path $asset.Source `
            -ExpectedHash $asset.ExpectedHash `
            -DisplayName $asset.Name
    }

    $expectedLauncherBatNames = @(
        '一键安装HABIT.bat',
        '一键启用HABIT-GPU.bat',
        '一键启用HABIT-AutoML.bat',
        '一键启用HABIT-进阶分析.bat',
        '启动HABIT命令行.bat'
    )
    $launcherDirectory = Join-Path $script:RepoRoot 'launchers'
    $launcherBatFiles = @(
        foreach ($batName in $expectedLauncherBatNames) {
            $batPath = Join-Path $launcherDirectory $batName
            if (-not (Test-Path -LiteralPath $batPath -PathType Leaf)) {
                throw "Missing required launcher BAT entry point: $batName"
            }
            Get-Item -LiteralPath $batPath
        }
    )
    $unexpectedLauncherBatFiles = @(
        Get-ChildItem -LiteralPath $launcherDirectory -File -Filter '*.bat' |
            Where-Object { $expectedLauncherBatNames -notcontains $_.Name }
    )
    if ($unexpectedLauncherBatFiles.Count -gt 0) {
        throw (
            "Unexpected launcher BAT files: " +
            ($unexpectedLauncherBatFiles.Name -join ', ')
        )
    }
    $rootBatFiles = @(Get-ChildItem -LiteralPath $script:RepoRoot -File -Filter '*.bat')
    if ($rootBatFiles.Count -gt 0) {
        throw (
            "Repository-root BAT files are not allowed; move user entry points to launchers: " +
            ($rootBatFiles.Name -join ', ')
        )
    }

    if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
        $OutputDirectory = Join-Path $script:RepoRoot 'dist'
    }
    elseif (-not [System.IO.Path]::IsPathRooted($OutputDirectory)) {
        $OutputDirectory = Join-Path $script:RepoRoot $OutputDirectory
    }
    $OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)

    # The repository-derived lock name prevents concurrent builds of this
    # checkout without unnecessarily blocking builds from unrelated checkouts.
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $repoBytes = [System.Text.Encoding]::UTF8.GetBytes(
            $script:RepoRoot.ToLowerInvariant()
        )
        $repoHash = ([System.BitConverter]::ToString(
                $sha.ComputeHash($repoBytes)
            )).Replace('-', '').Substring(0, 16)
    }
    finally {
        $sha.Dispose()
    }
    $lockPath = Join-Path ([System.IO.Path]::GetTempPath()) (
        "habit-lightweight-release-$repoHash.lock"
    )
    try {
        $lockStream = New-Object System.IO.FileStream(
            $lockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch {
        throw "Another lightweight release build is already running for this repository."
    }

    $temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
        "habit-lightweight-release-$([guid]::NewGuid().ToString('N'))"
    )
    $buildSource = Join-Path $temporaryRoot 'build-source'
    $wheelOutput = Join-Path $temporaryRoot 'wheel-output'
    $archiveRoot = Join-Path $temporaryRoot 'archive'
    $packageName = "HABIT-light-v$version"
    $packageRoot = Join-Path $archiveRoot $packageName
    New-Item -ItemType Directory -Path $buildSource, $wheelOutput, $packageRoot |
        Out-Null

    Write-Host '[HABIT] Building native wheel from an isolated source copy...'
    Copy-FilteredTree `
        -Source (Join-Path $script:RepoRoot 'habit') `
        -Destination (Join-Path $buildSource 'habit')
    foreach ($buildFileName in @(
            'setup.py',
            'pyproject.toml',
            'MANIFEST.in',
            'README.md',
            'LICENSE'
        )) {
        Copy-Item `
            -LiteralPath (Join-Path $script:RepoRoot $buildFileName) `
            -Destination (Join-Path $buildSource $buildFileName)
    }
    $optionalSetupCfg = Join-Path $script:RepoRoot 'setup.cfg'
    if (Test-Path -LiteralPath $optionalSetupCfg -PathType Leaf) {
        Copy-Item -LiteralPath $optionalSetupCfg -Destination $buildSource
    }

    $savedPythonPath = $env:PYTHONPATH
    $savedPythonHome = $env:PYTHONHOME
    $savedPipNoIndex = $env:PIP_NO_INDEX
    $savedPipDisableVersionCheck = $env:PIP_DISABLE_PIP_VERSION_CHECK
    try {
        $env:PYTHONPATH = $null
        $env:PYTHONHOME = $null
        $env:PIP_NO_INDEX = '1'
        $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
        Push-Location $buildSource
        try {
            & $resolvedPython `
                -I `
                -s `
                -m pip wheel . `
                --no-deps `
                --no-build-isolation `
                --no-cache-dir `
                --wheel-dir $wheelOutput
            if ($LASTEXITCODE -ne 0) {
                throw "HABIT wheel build failed with exit code $LASTEXITCODE."
            }
        }
        finally {
            Pop-Location
        }
    }
    finally {
        $env:PYTHONPATH = $savedPythonPath
        $env:PYTHONHOME = $savedPythonHome
        $env:PIP_NO_INDEX = $savedPipNoIndex
        $env:PIP_DISABLE_PIP_VERSION_CHECK = $savedPipDisableVersionCheck
    }

    $builtWheels = @(Get-ChildItem -LiteralPath $wheelOutput -File -Filter 'HABIT-*.whl')
    if ($builtWheels.Count -ne 1) {
        throw "Wheel build produced $($builtWheels.Count) HABIT wheels instead of one."
    }
    Assert-WheelContents -WheelPath $builtWheels[0].FullName

    Write-Host '[HABIT] Staging the release from the explicit allowlist...'
    foreach ($directoryName in @('habit', 'config', 'installer')) {
        Copy-FilteredTree `
            -Source (Join-Path $script:RepoRoot $directoryName) `
            -Destination (Join-Path $packageRoot $directoryName)
    }

    foreach ($requiredFileName in @(
            'LICENSE',
            'README.md',
            'pyproject.toml',
            'setup.py',
            'MANIFEST.in'
        )) {
        Copy-Item `
            -LiteralPath (Join-Path $script:RepoRoot $requiredFileName) `
            -Destination (Join-Path $packageRoot $requiredFileName)
    }
    foreach ($optionalRootFile in @(
            Get-ChildItem -LiteralPath $script:RepoRoot -File |
                Where-Object {
                    $_.Name -eq 'setup.cfg' -or
                    $_.Name -like 'README*'
                }
        )) {
        Copy-Item -LiteralPath $optionalRootFile.FullName -Destination $packageRoot -Force
    }
    $stagedLauncherDirectory = Join-Path $packageRoot 'launchers'
    New-Item -ItemType Directory -Path $stagedLauncherDirectory -Force | Out-Null
    foreach ($batFile in $launcherBatFiles) {
        Copy-Item -LiteralPath $batFile.FullName -Destination $stagedLauncherDirectory
    }

    New-Item `
        -ItemType Directory `
        -Path (Join-Path $packageRoot 'tools\wheels') `
        -Force |
        Out-Null
    Copy-Item `
        -LiteralPath $builtWheels[0].FullName `
        -Destination (Join-Path $packageRoot 'tools\wheels')

    foreach ($asset in $fixedAssets) {
        $destination = Join-Path $packageRoot ([string]$asset.Destination)
        New-Item `
            -ItemType Directory `
            -Path ([System.IO.Path]::GetDirectoryName($destination)) `
            -Force |
            Out-Null
        Copy-Item -LiteralPath $asset.Source -Destination $destination
        Assert-FileHash `
            -Path $destination `
            -ExpectedHash $asset.ExpectedHash `
            -DisplayName $asset.Name
    }

    $stagedLicenseDirectory = Join-Path $packageRoot 'licenses\micromamba'
    Copy-ExactTree `
        -Source $resolvedMicromambaLicenses `
        -Destination $stagedLicenseDirectory
    if (@(Get-ChildItem -LiteralPath $stagedLicenseDirectory -File -Recurse).Count -eq 0) {
        throw 'The staged micromamba license directory is empty.'
    }

    Write-ReleaseManifest -PackageRoot $packageRoot -Version $version
    Assert-FactoryChecks `
        -PackageRoot $packageRoot `
        -FixedAssets $fixedAssets `
        -LauncherBatEntries @(
            $launcherBatFiles | ForEach-Object { "launchers/$($_.Name)" }
        )

    Write-Host '[HABIT] Creating ZIP archive...'
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    $finalArchive = Join-Path $OutputDirectory "$packageName.zip"
    $temporaryArchive = Join-Path $temporaryRoot "$packageName.zip"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $archiveRoot,
        $temporaryArchive,
        [System.IO.Compression.CompressionLevel]::Optimal,
        $false
    )
    Move-Item -LiteralPath $temporaryArchive -Destination $finalArchive -Force

    Write-Host "[HABIT] Lightweight release created: $finalArchive"
}
finally {
    if ($null -ne $lockStream) {
        $lockStream.Dispose()
    }
    if ($null -ne $temporaryRoot -and (Test-Path -LiteralPath $temporaryRoot)) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
    if ($null -ne $lockPath -and (Test-Path -LiteralPath $lockPath)) {
        Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    }
}
