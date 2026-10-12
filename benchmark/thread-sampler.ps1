# Samples per-thread CPU times of the dedicated server, spaced apart, into $OutDir.
#
# The per-thread cpu=NNms counters in jcmd Thread.print are cumulative since thread start, so
# the DELTA between two samples divided by the spacing is that thread's busy fraction over the
# window. That is the instrument for "which thread is the wall" -- a rate tells us throughput,
# but only a busy fraction tells us which serial resource saturates as it moves.
#
# Usage: .\benchmark\thread-sampler.ps1 -Samples 4 -IntervalSeconds 10
param(
    [int]$Samples = 4,
    [int]$IntervalSeconds = 10,
    [string]$OutDir = "$env:TEMP\threadsamples"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Remove-Item "$OutDir\sample-*.txt" -ErrorAction SilentlyContinue

$java = Get-Process java | Sort-Object StartTime | Select-Object -Last 1
if ($null -eq $java) { Write-Error "no java process found"; exit 1 }
"pid=$($java.Id) samples=$Samples interval=${IntervalSeconds}s"

for ($i = 0; $i -lt $Samples; $i++) {
    & jcmd $java.Id Thread.print 2>$null | Out-File -Encoding utf8 "$OutDir\sample-$i.txt"
    "sample $i saved ($((Get-Item "$OutDir\sample-$i.txt").Length) bytes)"
    if ($i -lt $Samples - 1) { Start-Sleep -Seconds $IntervalSeconds }
}
