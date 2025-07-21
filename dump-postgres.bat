@echo off
setlocal

REM Load environment variables
for /f "tokens=1,2 delims==" %%A in ('type .env') do set %%A=%%B

REM Dump PostgreSQL database from container
echo Dumping PostgreSQL data from container...
docker exec employee-postgres pg_dump -U %POSTGRES_USER% -d %POSTGRES_DB% -F p > employee-sample-database/postgres/init.sql

echo Done. The updated init.sql file now contains the latest schema and data.
endlocal
pause
