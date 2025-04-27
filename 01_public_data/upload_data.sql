CREATE DATABASE test1;

CREATE SCHEMA IF NOT EXISTS TestSchema;

CREATE TABLE IF NOT EXISTS TestSchema.encounters (
    Id VARCHAR(50),
    START TIMESTAMP,
    STOP TIMESTAMP,
    PATIENT VARCHAR(50),
    ORGANIZATION VARCHAR(100),
    PAYER VARCHAR(100),
    ENCOUNTERCLASS VARCHAR(50),
    CODE VARCHAR(50),
    DESCRIPTION TEXT,
    BASE_ENCOUNTER_COST NUMERIC(10,2),
    TOTAL_CLAIM_COST NUMERIC(10,2),
    PAYER_COVERAGE NUMERIC(10,2),
    REASONCODE VARCHAR(50),
    REASONDESCRIPTION TEXT
);

-- psql client command (must run from a psql terminal on your local machine)
\COPY TestSchema.encounters FROM '.....01_public_data/encounters_modified.csv' DELIMITER ',' CSV HEADER;

