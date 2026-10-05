# Drives the dedicated server console over RCON, one command per invocation.
#
# The headless benchmark harness: with enable-rcon=true in run/server.properties, this lets a
# session start a Chunky pre-generation run, wait, and cancel it, without a player or a terminal.
#
# Usage:
#   .\benchmark\chunky-rcon.ps1 -Command "chunky start"
#   .\benchmark\chunky-rcon.ps1 -Command "stop"
param(
    [Parameter(Mandatory = $true)][string]$Command,
    [string]$ServerHost = "127.0.0.1",
    [int]$Port = 25575,
    [string]$Password = "terracuda",
    [int]$TimeoutSeconds = 15
)

$ErrorActionPreference = "Stop"

$client = New-Object System.Net.Sockets.TcpClient
$client.Connect($ServerHost, $Port)
$stream = $client.GetStream()
$stream.ReadTimeout = $TimeoutSeconds * 1000
$stream.WriteTimeout = $TimeoutSeconds * 1000

function Write-RconPacket([int]$id, [int]$type, [string]$payload) {
    $bytes = [System.Text.Encoding]::ASCII.GetBytes($payload)
    $length = 4 + 4 + $bytes.Length + 2
    $buffer = New-Object System.IO.MemoryStream
    $writer = New-Object System.IO.BinaryWriter($buffer)
    $writer.Write([int32]$length)
    $writer.Write([int32]$id)
    $writer.Write([int32]$type)
    $writer.Write($bytes)
    $writer.Write([byte]0)
    $writer.Write([byte]0)
    $writer.Flush()
    $data = $buffer.ToArray()
    $stream.Write($data, 0, $data.Length)
    $stream.Flush()
}

function Read-RconPacket() {
    $header = New-Object byte[] 12
    $read = 0
    while ($read -lt 12) {
        $chunk = $stream.Read($header, $read, 12 - $read)
        if ($chunk -le 0) { return $null }
        $read += $chunk
    }
    $length = [BitConverter]::ToInt32($header, 0)
    $id = [BitConverter]::ToInt32($header, 4)
    $type = [BitConverter]::ToInt32($header, 8)
    $bodyLength = [Math]::Max(0, $length - 8)
    $body = New-Object byte[] $bodyLength
    $read = 0
    while ($read -lt $bodyLength) {
        $chunk = $stream.Read($body, $read, $bodyLength - $read)
        if ($chunk -le 0) { break }
        $read += $chunk
    }
    @{ Id = $id; Type = $type; Body = ([System.Text.Encoding]::ASCII.GetString($body)).Trim([char]0) }
}

try {
    # Source RCON auth: the server answers with a value packet then an auth response; id -1 = denied.
    Write-RconPacket 1 3 $Password
    while ($true) {
        $packet = Read-RconPacket
        if ($packet -eq $null) { Write-Error "connection closed during auth"; return }
        if ($packet.Type -eq 2) {
            if ($packet.Id -eq -1) { Write-Error "rcon authentication failed (wrong password?)"; return }
            break
        }
    }

    Write-RconPacket 2 2 $Command
    $response = Read-RconPacket
    if ($response -ne $null) {
        $response.Body
    } else {
        Write-Output "(no response)"
    }
} finally {
    $stream.Close()
    $client.Close()
}
