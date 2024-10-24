-- Adjust PostgreSQL configuration parameters
-- DB Version: 17
-- OS Type: linux
-- DB Type: dw
-- Total Memory (RAM): 4 GB
-- Data Storage: ssd
-- SOURCE: https://pgtune.leopard.in.ua/

ALTER SYSTEM SET max_connections = 40;
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET effective_cache_size = '3GB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 500;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '6553kB';
ALTER SYSTEM SET huge_pages = 'off';
ALTER SYSTEM SET min_wal_size = '4GB';
ALTER SYSTEM SET max_wal_size = '16GB';
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 0;
ALTER SYSTEM SET log_connections = 'on';
ALTER SYSTEM SET log_disconnections = 'on';
ALTER SYSTEM SET log_error_verbosity = 'verbose';
ALTER SYSTEM SET log_destination = 'stderr';

-- Reload the PostgreSQL configuration
SELECT pg_reload_conf();
