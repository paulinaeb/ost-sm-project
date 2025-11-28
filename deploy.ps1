#requires -Version 5.1
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Config
$COMPOSE_FILE = 'docker-compose.yml'
$LINKEDIN_KEYSPACE = 'linkedin_jobs'
$LINKEDIN_TABLE = 'jobs'
$CASSANDRA_CONTAINER = 'cassandra-dev'
$KAFKA_CONTAINER = 'kafka'
$APP_CONTAINER = 'python-app'
$STREAMLIT_URL = 'http://localhost:8501'
$SCHEMA_FILE = 'preprocessing/ECSF/keyspace_tables_creation.sql'
$ECSF_KEYSPACE = 'ecsf'

function Test-Command {
    param([Parameter(Mandatory)][string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Resolve-ComposeMode {
    if (Test-Command -Name 'docker') {
        try { docker compose version | Out-Null; $script:UseComposeV2 = $true; return } catch {}
    }
    if (Test-Command -Name 'docker-compose') { $script:UseComposeV2 = $false; return }
    throw 'ERROR: docker compose not installed.'
}

function Invoke-Compose {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
    if ($script:UseComposeV2) {
        & docker compose @Args
    } else {
        & docker-compose @Args
    }
}

function Wait-Health {
    param(
        [Parameter(Mandatory)][string]$Name,
        [int]$TimeoutSeconds = 300
    )
    Write-Host "Waiting for $Name to be healthy (timeout ${TimeoutSeconds}s)..."
    $elapsed = 0
    while ($true) {
        try {
            $status = docker inspect -f '{{.State.Health.Status}}' $Name 2>$null
            if ($status -eq 'healthy') { break }
        } catch {}
        Start-Sleep -Seconds 5
        $elapsed += 5
        if ($elapsed -ge $TimeoutSeconds) {
            Write-Error "ERROR: $Name did not become healthy in time."
            try { docker logs $Name --tail 50 | Write-Host } catch {}
            throw "Timeout waiting for $Name health."
        }
    }
    Write-Host "$Name is healthy."
}

function Get-FirstNumberOrDefault {
    param(
        [string]$Text,
        [int]$Default = 0
    )
    $m = [regex]::Match($Text, '\\d+')
    if ($m.Success) { return [int]$m.Value } else { return $Default }
}

try {
    Write-Host '[1/7] Prechecks...'
    if (-not (Test-Command -Name 'docker')) { throw 'ERROR: docker not found.' }
    docker info | Out-Null
    Resolve-ComposeMode
    if (-not (Test-Path $COMPOSE_FILE)) { throw "ERROR: $COMPOSE_FILE not found." }
    if (-not (Test-Path 'requirements.txt')) { throw 'ERROR: requirements.txt not found.' }
    if (-not (Test-Path 'forecasting/streamlit_app.py')) { throw 'ERROR: forecasting/streamlit_app.py not found.' }

    Write-Host '[2/7] Building app image...'
    Invoke-Compose -Args @('-f', $COMPOSE_FILE, 'build', 'python-app')

    Write-Host '[3/7] Pulling external images...'
    Invoke-Compose -Args @('-f', $COMPOSE_FILE, 'pull', 'cassandra-dev', 'kafka', 'kafka-ui', 'cassandra-web')

    Write-Host '[4/7] Starting services...'
    Invoke-Compose -Args @('-f', $COMPOSE_FILE, 'up', '-d', 'cassandra-dev', 'kafka', 'kafka-ui', 'cassandra-web', 'python-app')

    Write-Host '[5/7] Waiting for dependencies...'
    Wait-Health -Name $CASSANDRA_CONTAINER -TimeoutSeconds 300
    Wait-Health -Name $KAFKA_CONTAINER -TimeoutSeconds 300

    Write-Host '[6/7] Initializing Cassandra schema (ECSF keyspace)...'
    $keyspaceExists = $false
    $ksCheckOut = ''
    try {
        $ksCheckOut = docker exec -i $CASSANDRA_CONTAINER cqlsh -e "DESCRIBE KEYSPACE $ECSF_KEYSPACE"
        if ($LASTEXITCODE -eq 0) { $keyspaceExists = $true } else { $keyspaceExists = $false }
    } catch { $keyspaceExists = $false }

    if ($keyspaceExists) {
        Write-Host "Keyspace '$ECSF_KEYSPACE' already exists."
        # If keyspace exists but required tables don't, initialize schema
        $tableCheckOut = ''
        $tablesOk = $false
        try {
            $tableCheckOut = docker exec -i $CASSANDRA_CONTAINER cqlsh -e "DESCRIBE TABLE $ECSF_KEYSPACE.work_role_by_id"
            if ($LASTEXITCODE -eq 0) { $tablesOk = $true } else { $tablesOk = $false }
        } catch { $tablesOk = $false }
        if (-not $tablesOk) {
            Write-Host 'ECSF tables missing. Initializing schema...'
            if (Test-Path $SCHEMA_FILE) {
                $attempt = 1
                while ($attempt -le 3) {
                    Write-Host "Attempting schema init (try $attempt/3)..."
                    try {
                        Get-Content -Raw $SCHEMA_FILE | docker exec -i $CASSANDRA_CONTAINER cqlsh | Out-Null
                        if ($LASTEXITCODE -eq 0) {
                            Write-Host "Schema initialized from $SCHEMA_FILE."
                            break
                        } else { throw "Schema init failed with exit code $LASTEXITCODE" }
                    } catch {
                        if ($attempt -lt 3) { Write-Host 'Schema init failed, retrying in 5s...'; Start-Sleep -Seconds 5 }
                        else { throw 'ERROR: Schema init failed after 3 attempts.' }
                    }
                    $attempt++
                }
            } else {
                Write-Warning "WARNING: $SCHEMA_FILE not found. Skipping schema init."
            }
        }
    } else {
        if (Test-Path $SCHEMA_FILE) {
            Write-Host 'Waiting for Cassandra to be fully ready...'
            Start-Sleep -Seconds 10
            $attempt = 1
            while ($attempt -le 3) {
                Write-Host "Attempting schema init (try $attempt/3)..."
                try {
                    Get-Content -Raw $SCHEMA_FILE | docker exec -i $CASSANDRA_CONTAINER cqlsh | Out-Null
                    if ($LASTEXITCODE -eq 0) {
                        Write-Host "Schema initialized from $SCHEMA_FILE."
                        break
                    } else {
                        throw "Schema init failed with exit code $LASTEXITCODE"
                    }
                } catch {
                    if ($attempt -lt 3) {
                        Write-Host 'Schema init failed, retrying in 5s...'
                        Start-Sleep -Seconds 5
                    } else {
                        throw 'ERROR: Schema init failed after 3 attempts.'
                    }
                }
                $attempt++
            }
        } else {
            Write-Warning "WARNING: $SCHEMA_FILE not found. Skipping schema init."
        }
    }

    Write-Host 'Checking if ECSF data already loaded...'
    $ROW_COUNT = 0
    try {
        # Use -k keyspace to avoid DESCRIBE/USE output noise and get clean COUNT
        $ecsfCountOut = docker exec -i $CASSANDRA_CONTAINER cqlsh -k $ECSF_KEYSPACE --no-color -e "SELECT COUNT(*) FROM work_role_by_id;"
        if ($LASTEXITCODE -eq 0) {
            # Extract a standalone integer line (robust against headers)
            $m = [regex]::Match($ecsfCountOut, '(?m)^\s*(\d+)\s*$')
            if ($m.Success) { $ROW_COUNT = [int]$m.Groups[1].Value } else { $ROW_COUNT = Get-FirstNumberOrDefault -Text $ecsfCountOut -Default 0 }
        } else { $ROW_COUNT = 0 }
    } catch { $ROW_COUNT = 0 }

    Write-Host "ECSF row count detected: $ROW_COUNT"
    if ($ROW_COUNT -le 3) {
        # Fallback: try selecting a single row to detect presence
        $hasAnyRow = $false
        try {
            $oneRowOut = docker exec -i $CASSANDRA_CONTAINER cqlsh --no-color -e "SELECT work_role_id FROM $ECSF_KEYSPACE.work_role_by_id LIMIT 1"
            if ($LASTEXITCODE -eq 0 -and $oneRowOut -match '\\d') { $hasAnyRow = $true }
        } catch { $hasAnyRow = $false }
        if ($hasAnyRow) {
            Write-Host 'ECSF data detected via LIMIT 1 fallback. Skipping load.'
        } else {
        Write-Host 'Few ECSF rows detected (<=3). Loading initial dataset...'
        $cid = (Invoke-Compose -Args @('-f', $COMPOSE_FILE, 'ps', '-q', 'cassandra-dev'))
        $netId = (docker inspect -f '{{range .NetworkSettings.Networks}}{{.NetworkID}}{{end}}' $cid)
        $NETWORK_NAME = (docker network inspect -f '{{.Name}}' $netId)
        docker run --rm --network $NETWORK_NAME `
            -e CASSANDRA_HOSTS=cassandra-dev `
            -e CASSANDRA_PORT=9042 `
            -e CASSANDRA_KEYSPACE=ecsf `
            csoma-streamlit:latest `
            python preprocessing/ECSF/load_ecsf.py | Write-Host
        Write-Host 'ECSF data loaded.'
        }
    } else {
        Write-Host 'ECSF data already present. Skipping load.'
    }

    Write-Host 'Checking if LinkedIn jobs data already loaded...'
    # Ensure LinkedIn keyspace exists (even if table doesn't)
    $lkExists = $false
    try {
        docker exec -i $CASSANDRA_CONTAINER cqlsh -e "DESCRIBE KEYSPACE $LINKEDIN_KEYSPACE" | Out-Null
        if ($LASTEXITCODE -eq 0) { $lkExists = $true }
    } catch { $lkExists = $false }
    if (-not $lkExists) {
        Write-Host "Creating keyspace '$LINKEDIN_KEYSPACE'..."
        docker exec -i $CASSANDRA_CONTAINER cqlsh -e "CREATE KEYSPACE IF NOT EXISTS $LINKEDIN_KEYSPACE WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};" | Out-Null
        Start-Sleep -Seconds 2
    }

    # Ensure LinkedIn jobs table exists explicitly (matches consumer schema)
    $jobsTableExists = $false
    try {
        docker exec -i $CASSANDRA_CONTAINER cqlsh -e "DESCRIBE TABLE $LINKEDIN_KEYSPACE.$LINKEDIN_TABLE" | Out-Null
        if ($LASTEXITCODE -eq 0) { $jobsTableExists = $true } else { $jobsTableExists = $false }
    } catch { $jobsTableExists = $false }
    if (-not $jobsTableExists) {
        Write-Host "Creating table '$LINKEDIN_KEYSPACE.$LINKEDIN_TABLE'..."
        $createJobsCql = @"
CREATE TABLE IF NOT EXISTS $LINKEDIN_KEYSPACE.$LINKEDIN_TABLE (
    id int PRIMARY KEY,
    title text,
    primary_description text,
    detail_url text,
    location text,
    skill text,
    company_name text,
    created_at text,
    country text,
    ingested_at timestamp
);
"@
        $createJobsCql | docker exec -i $CASSANDRA_CONTAINER cqlsh | Out-Null
        Start-Sleep -Seconds 2
        # re-check
        try {
            docker exec -i $CASSANDRA_CONTAINER cqlsh -e "DESCRIBE TABLE $LINKEDIN_KEYSPACE.$LINKEDIN_TABLE" | Out-Null
            if ($LASTEXITCODE -eq 0) { $jobsTableExists = $true } else { $jobsTableExists = $false }
        } catch { $jobsTableExists = $false }
    }
    # Now decide based on row count
    $countOut = ''
    try {
        $countOut = docker exec -i $CASSANDRA_CONTAINER cqlsh -k $LINKEDIN_KEYSPACE --no-color -e "SELECT COUNT(*) FROM $LINKEDIN_TABLE;"
    } catch { $countOut = '' }
    if ($LASTEXITCODE -eq 0) {
        $m2 = [regex]::Match($countOut, '(?m)^\s*(\d+)\s*$')
        if ($m2.Success) { $COUNT_OUTPUT = [int]$m2.Groups[1].Value } else { $COUNT_OUTPUT = Get-FirstNumberOrDefault -Text $countOut -Default 0 }
    } else { $COUNT_OUTPUT = 0 }
    Write-Host "LinkedIn jobs row count detected: $COUNT_OUTPUT"
    if ($COUNT_OUTPUT -le 3) {
        Write-Host 'Few LinkedIn rows detected (<=3). Starting streaming pipeline...'
        Invoke-Compose -Args @('-f', $COMPOSE_FILE, 'up', '-d', 'kafka-consumer')
        Write-Host 'Waiting for Consumer to be fully ready...'
        Start-Sleep -Seconds 10
        Invoke-Compose -Args @('-f', $COMPOSE_FILE, 'up', '-d', 'kafka-producer')
        if ((docker inspect -f '{{.State.ExitCode}}' 'kafka-producer') -ne '0') {
            docker logs 'kafka-producer' | Write-Host
            throw 'ERROR: Producer failed.'
        }
        if ((docker inspect -f '{{.State.ExitCode}}' 'kafka-consumer') -ne '0') {
            docker logs 'kafka-consumer' | Write-Host
            throw 'ERROR: Consumer failed.'
        }
    } else {
        Write-Host "LinkedIn jobs data already present ($COUNT_OUTPUT rows). Skipping streaming pipeline."
    }

    Write-Host '[7/7] Waiting for Streamlit app...'
    Wait-Health -Name $APP_CONTAINER -TimeoutSeconds 300

    Write-Host ''
    Write-Host 'All services are up:'
    Write-Host "- Streamlit: $STREAMLIT_URL"
    Write-Host '- Kafka UI: http://localhost:8080'
    Write-Host '- Cassandra Web: http://localhost:8081'
    Invoke-Compose -Args @('-f', $COMPOSE_FILE, 'ps')
}
catch {
    Write-Error $_
    exit 1
}
