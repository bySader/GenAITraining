# SQL Basics Guide

## Author: Sandra Kim
## Topic: Databases

SQL (Structured Query Language) is the standard language for managing and querying relational databases. It allows users to create, read, update, and delete data stored in tables.

## Core SQL Commands

### Data Query Language (DQL)

```sql
-- Select all columns from a table
SELECT * FROM employees;

-- Select specific columns with a filter
SELECT first_name, last_name, salary
FROM employees
WHERE department = 'Engineering'
ORDER BY salary DESC;

-- Aggregate functions
SELECT department, COUNT(*) AS employee_count, AVG(salary) AS avg_salary
FROM employees
GROUP BY department
HAVING COUNT(*) > 5;
```

### Data Manipulation Language (DML)

```sql
-- Insert a new record
INSERT INTO employees (first_name, last_name, department, salary)
VALUES ('Alice', 'Smith', 'Engineering', 95000);

-- Update an existing record
UPDATE employees
SET salary = 100000
WHERE employee_id = 42;

-- Delete a record
DELETE FROM employees
WHERE employee_id = 42;
```

### Data Definition Language (DDL)

```sql
-- Create a table
CREATE TABLE employees (
    employee_id   INT PRIMARY KEY AUTO_INCREMENT,
    first_name    VARCHAR(50) NOT NULL,
    last_name     VARCHAR(50) NOT NULL,
    department    VARCHAR(100),
    salary        DECIMAL(10, 2),
    hire_date     DATE
);

-- Add a column
ALTER TABLE employees ADD COLUMN email VARCHAR(255);

-- Drop a table
DROP TABLE IF EXISTS temp_data;
```

## JOINs

JOINs combine rows from two or more tables based on a related column.

```sql
-- INNER JOIN: returns rows with matching values in both tables
SELECT e.first_name, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.id;

-- LEFT JOIN: returns all rows from left table, matched rows from right
SELECT e.first_name, p.project_name
FROM employees e
LEFT JOIN projects p ON e.employee_id = p.lead_employee_id;
```

## Indexes

Indexes speed up query performance by creating a data structure that allows the database to find rows without scanning the entire table.

```sql
CREATE INDEX idx_department ON employees(department);
CREATE UNIQUE INDEX idx_email ON employees(email);
```

## Constraints

```sql
-- Primary Key: unique identifier for each row
-- Foreign Key: enforces referential integrity
CREATE TABLE orders (
    order_id    INT PRIMARY KEY,
    customer_id INT NOT NULL,
    total       DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

## Transactions

Transactions ensure data integrity by grouping multiple operations into an atomic unit.

```sql
BEGIN TRANSACTION;
    UPDATE accounts SET balance = balance - 500 WHERE account_id = 1;
    UPDATE accounts SET balance = balance + 500 WHERE account_id = 2;
COMMIT;  -- or ROLLBACK if an error occurs
```

## Popular SQL Databases
- **PostgreSQL**: Open-source, feature-rich, great for complex queries
- **MySQL / MariaDB**: Widely used for web applications
- **SQLite**: Lightweight, file-based, great for development and mobile apps
- **Microsoft SQL Server**: Enterprise-grade, tight Windows integration
