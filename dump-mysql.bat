@echo off
setlocal

REM Load environment variables
for /f "tokens=1,2 delims==" %%A in ('type .env') do set %%A=%%B

REM Dump MySQL database from running container to local file
echo Dumping MySQL data from container...
docker exec employee-mysql mysqldump -u root -p%MYSQL_ROOT_PASSWORD% --databases %MYSQL_DATABASE% > employee-sample-database/mysql/init.sql

echo Done. The updated init.sql file now contains the latest schema and data.
endlocal
pause
