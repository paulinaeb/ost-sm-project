## 📊 Datasets

| Dataset | Description |
|--------|-------------|
| **LinkedIn Job Ads** | Raw job postings |
| **ECSF Framework** | Official European Cybersecurity Skills Framework |

### ECSF consists of 3 core tables:
| Table | Purpose |
|-------|--------|
| `work_role` | Defines cybersecurity job roles |
| `tks` | Defines **Tasks**, **Knowledge**, and **Skills** |
| `relationship` | Links roles ↔ required TKS elements |

> All data is pre-processed and loaded into **denormalized Cassandra tables** for fast querying.

---

## ⚙️ System Requirements

- **OS**: Windows 10/11
- **Tools**:
  - Docker Desktop
  - Python 3.9+
  - Git

---

## 🚀 Step-by-Step Setup & Execution

> **All commands below are for Windows**

---

### 1. Pre-process ECSF Dataset 🧹

> File: `ECSF/clean_ads_and_merge.ipynb`

#### Tasks:
1. **Inspect & Validate JSON**
   - Ensure required fields: `id`, `title`, `tks[].id`, etc.
   - Enforce **uniqueness**: `work_role.id`, `tks.id`
   - Check **referential integrity**:
     - Every `relationship.tks_id` exists in `tks`
     - Every `relationship.work_role_id` exists in `work_role`

2. **Clean & Normalize**
   - Trim whitespace, lowercase strings
   - Standardize IDs:
     - `work_role.id` → `int`
     - `tks.id` → format like `K0001`
   - Remove duplicates (same title + TKS)
   - Convert types: dates → ISO, booleans, integers
   - Standardize lists: `alternative_title(s)` → clean array

> Output: Clean CSV files in `ECSF/data/`:
> - `work_role_by_id.csv`
> - `role_with_tks.csv`
> - `roles_by_title.csv`
> - `roles_by_tks.csv`
> - `tks_by_id.csv`

---

### 2. Start Cassandra with Docker 🐳

```bash
# Pull and run Cassandra
docker run --name cassandra-dev -d -p 9042:9042 -p 9160:9160 cassandra:4.1
```

> Wait **20–30 seconds** for startup  
> Check logs:
```bash
docker logs -f cassandra-dev
```

**Connection Details**:
- Host: `127.0.0.1`
- Port: `9042` (CQL native)
- **Not** HTTPS!

**Stop & Remove**:
```bash
docker stop cassandra-dev && docker rm cassandra-dev
```

---

### 3. Create Keyspace & Tables in Cassandra 📋

```bash
# Open interactive CQL shell
docker exec -it cassandra-dev cqlsh
```

Inside `cqlsh`, run:
```sql
SOURCE 'ECSF/keyspace_tables_creation.sql';
```

> This creates:
> - Keyspace: `ecsf`
> - Denormalized tables optimized for query patterns (by ID, title, TKS)

---

### 4. Load Data into Cassandra 📥

#### 4.1 Set up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate.bat

# Install dependencies
pip install cassandra-driver pandas
```

#### 4.2 Run Data Loader

> Place `load_ecsf.py` in the same folder as the **CSV files**

```bash
python load_ecsf.py
```

> Loads all CSVs into respective Cassandra tables using batch inserts

---

### 5. Verify & Test Queries ✅

```bash
docker exec -it cassandra-dev cqlsh
```

Inside `cqlsh`:
```sql
USE ecsf;

-- Count total roles
SELECT count(*) FROM work_role_by_id;

-- Search role by title (case-insensitive)
SELECT work_role_id, canonical_title 
FROM roles_by_title 
WHERE title_key = 'chief information security officer (ciso)';
```

---

## 📈 Query Patterns Supported

| Use Case | Table to Query |
|--------|----------------|
| Get role by ID | `work_role_by_id` |
| Get role + all TKS | `role_with_tks` |
| Search role by title | `roles_by_title` |
| Find roles requiring a TKS | `roles_by_tks` |
| Get TKS details | `tks_by_id` |

---
