-- create keyspace (dev)
CREATE KEYSPACE IF NOT EXISTS ecsf WITH replication = {'class':'SimpleStrategy', 'replication_factor':1};

USE ecsf;

-- work_role_by_id
CREATE TABLE IF NOT EXISTS work_role_by_id (
  work_role_id int PRIMARY KEY,
  title text,
  alt_titles list<text>,
  summary_statement text,
  mission text,
  tks_ids list<text>,
  metadata map<text,text>
);

-- role_with_tks
CREATE TABLE IF NOT EXISTS role_with_tks (
  work_role_id int PRIMARY KEY,
  title text,
  alt_titles list<text>,
  summary_statement text,
  mission text,
  tks list<frozen<tuple<text,text,text>>>
);

-- roles_by_title
CREATE TABLE IF NOT EXISTS roles_by_title (
  title_key text PRIMARY KEY,
  work_role_id int,
  is_alt boolean,
  canonical_title text
);

-- roles_by_tks
CREATE TABLE IF NOT EXISTS roles_by_tks (
  tks_id text,
  work_role_id int,
  PRIMARY KEY (tks_id, work_role_id)
);

-- tks_by_id
CREATE TABLE IF NOT EXISTS tks_by_id (
  tks_id text PRIMARY KEY,
  type text,
  description text
);
