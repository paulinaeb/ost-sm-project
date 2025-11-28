#requires -Version 5.1
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Config
$COMPOSE_FILE = 'docker-compose.yml'
$CASSANDRA_CONTAINER = 'cassandra-dev'
$LINKEDIN_KEYSPACE = 'linkedin_jobs'
$LINKEDIN_TABLE = 'jobs'

function Test-Command {
    param([Parameter(Mandatory)][string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# Detect Compose v2 vs v1 and provide a unified invoker
$script:UseComposeV2 = $false
function Resolve-ComposeMode {
    $script:UseComposeV2 = $false
    if (Test-Command -Name 'docker') {
        try { docker compose version | Out-Null; $script:UseComposeV2 = $true } catch { $script:UseComposeV2 = $false }
    }
    if (-not $script:UseComposeV2) {
        if (Test-Command -Name 'docker-compose') { $script:UseComposeV2 = $false }
        else { throw 'ERROR: docker compose not installed.' }
    }
}
function Invoke-Compose {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
    if ($script:UseComposeV2) { & docker compose @Args } else { & docker-compose @Args }
}

try {
    Write-Host '[1/4] Prechecks...'
    if (-not (Test-Command -Name 'docker')) { throw 'ERROR: docker not found.' }
    docker info | Out-Null
    Resolve-ComposeMode
    if (-not (Test-Path $COMPOSE_FILE)) { throw "ERROR: $COMPOSE_FILE not found." }

    Write-Host '[2/4] Truncating LinkedIn jobs data...'
    try {
        docker exec -i $CASSANDRA_CONTAINER cqlsh -e "TRUNCATE $LINKEDIN_KEYSPACE.$LINKEDIN_TABLE" | Out-Null
        Write-Host 'LinkedIn jobs table truncated successfully.'
    } catch {
        Write-Warning 'WARNING: Could not truncate table. It may not exist yet.'
    }

    Write-Host '[3/4] Starting Kafka consumer...'
    Invoke-Compose -Args @('-f', $COMPOSE_FILE, 'up', '-d', 'kafka-consumer')
    Write-Host 'Waiting for Consumer to be fully ready...'
    Start-Sleep -Seconds 7

    Write-Host '[4/4] Starting Kafka producer...'
    Invoke-Compose -Args @('-f', $COMPOSE_FILE, 'up', '-d', 'kafka-producer')

    Write-Host ''
    Write-Host 'Stream mining process restarted successfully!'
    Write-Host ''
    Write-Host 'To monitor the streaming process:'
    Write-Host '  - Consumer logs: docker logs -f kafka-consumer'
    Write-Host '  - Producer logs: docker logs -f kafka-producer'
    Write-Host '  - Kafka UI: http://localhost:8080'
}
catch {
    Write-Error $_
    exit 1
}
