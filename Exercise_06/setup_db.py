"""
Exercise 06 -- Database Setup Script
Creates and seeds the SQLite database with 5 tables and mock data.
Run once: python setup_db.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "company.db"


def setup():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ── TABLE: TICKETS ────────────────────────────────────────────────────────
    c.execute("DROP TABLE IF EXISTS tickets")
    c.execute("""
        CREATE TABLE tickets (
            ticket_id    TEXT PRIMARY KEY,
            title        TEXT NOT NULL,
            priority     TEXT NOT NULL,   -- Critical, High, Medium, Low
            status       TEXT NOT NULL,   -- Open, In Progress, Resolved, Closed
            owner        TEXT NOT NULL,
            owner_id     TEXT NOT NULL,
            project_id   TEXT,
            created_date TEXT NOT NULL
        )
    """)
    tickets = [
        ("TKT-001","VPN connection failure",       "Critical","Open",       "Alice Martinez","EMP-001","PRJ-001","2026-08-01"),
        ("TKT-002","Outlook sync issue",            "High",    "In Progress","Bob Johnson",   "EMP-002","PRJ-002","2026-08-05"),
        ("TKT-003","Laptop screen flickering",      "Low",     "Resolved",   "Alice Martinez","EMP-001","PRJ-001","2026-07-25"),
        ("TKT-004","Cannot access shared drive",    "High",    "Open",       "Alice Martinez","EMP-001","PRJ-001","2026-08-10"),
        ("TKT-005","Printer offline on floor 3",    "Medium",  "Open",       "David Chen",    "EMP-004","PRJ-003","2026-08-12"),
        ("TKT-006","MFA not working for new hire",  "Critical","Open",       "Emma Rodriguez","EMP-005","PRJ-002","2026-08-15"),
        ("TKT-007","Database backup failure",       "Critical","In Progress","David Chen",    "EMP-004","PRJ-003","2026-08-18"),
        ("TKT-008","Slow internet in meeting rooms","Medium",  "Open",       "Carol Williams","EMP-003","PRJ-004","2026-08-03"),
        ("TKT-009","Software license expired",      "High",    "Open",       "Bob Johnson",   "EMP-002","PRJ-002","2026-08-08"),
        ("TKT-010","Onboarding portal 500 error",   "High",    "In Progress","Emma Rodriguez","EMP-005","PRJ-005","2026-08-19"),
        ("TKT-011","Cloud storage quota exceeded",  "Medium",  "Open",       "Carol Williams","EMP-003","PRJ-004","2026-08-14"),
        ("TKT-012","Docker build pipeline failing", "Critical","Open",       "David Chen",    "EMP-004","PRJ-003","2026-08-20"),
    ]
    c.executemany("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?)", tickets)

    # ── TABLE: EMPLOYEES ──────────────────────────────────────────────────────
    c.execute("DROP TABLE IF EXISTS employees")
    c.execute("""
        CREATE TABLE employees (
            employee_id TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            department  TEXT NOT NULL,
            hire_date   TEXT NOT NULL,
            project     TEXT,
            skills      TEXT          -- comma-separated list
        )
    """)
    employees = [
        ("EMP-001","Alice Martinez", "IT",        "2021-03-15","Phoenix",    "Python,DevOps,Kubernetes"),
        ("EMP-002","Bob Johnson",    "HR",         "2020-06-01","Orion",      "HR Systems,SAP,Onboarding"),
        ("EMP-003","Carol Williams", "Finance",    "2019-09-10","Atlas",      "Excel,PowerBI,SQL"),
        ("EMP-004","David Chen",     "IT",         "2022-01-20","Phoenix",    "Docker,Kubernetes,CI/CD"),
        ("EMP-005","Emma Rodriguez", "Marketing",  "2023-04-05","Nova",       "SEO,Google Analytics,Ads"),
        ("EMP-006","Frank Lee",      "IT",         "2020-11-30","Atlas",      "Python,Machine Learning,SQL"),
        ("EMP-007","Grace Kim",      "Engineering","2021-07-22","Phoenix",    "Java,Spring Boot,Microservices"),
        ("EMP-008","Henry Brown",    "Sales",      "2022-05-14","Orion",      "CRM,Salesforce,Negotiation"),
        ("EMP-009","Iris Patel",     "IT",         "2023-08-01","Nova",       "React,TypeScript,GraphQL"),
        ("EMP-010","James Wilson",   "Engineering","2018-02-28","Atlas",      "C++,Embedded Systems,RTOS"),
    ]
    c.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?)", employees)

    # ── TABLE: PROJECTS ───────────────────────────────────────────────────────
    c.execute("DROP TABLE IF EXISTS projects")
    c.execute("""
        CREATE TABLE projects (
            project_id        TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            duration          TEXT,
            team              TEXT,   -- comma-separated employee IDs
            termination_date  TEXT,
            has_open_tickets  INTEGER DEFAULT 0,
            open_tickets      INTEGER DEFAULT 0
        )
    """)
    projects = [
        ("PRJ-001","Phoenix",         "18 months","EMP-001,EMP-004,EMP-007","2027-02-28",1,3),
        ("PRJ-002","Orion",           "12 months","EMP-002,EMP-005,EMP-008","2026-12-31",1,2),
        ("PRJ-003","Atlas",           "24 months","EMP-003,EMP-004,EMP-006","2027-06-30",1,2),
        ("PRJ-004","Nova",            "6 months", "EMP-005,EMP-009",       "2026-09-30",1,2),
        ("PRJ-005","Nebula",          "9 months", "EMP-002,EMP-010",       "2026-11-15",1,1),
        ("PRJ-006","Titan",           "15 months","EMP-006,EMP-007,EMP-010","2027-03-31",0,0),
        ("PRJ-007","Polaris",         "3 months", "EMP-008,EMP-009",       "2026-09-01",0,0),
    ]
    c.executemany("INSERT INTO projects VALUES (?,?,?,?,?,?,?)", projects)

    # ── TABLE: CLIENTS ────────────────────────────────────────────────────────
    c.execute("DROP TABLE IF EXISTS clients")
    c.execute("""
        CREATE TABLE clients (
            client_id       TEXT PRIMARY KEY,
            client_name     TEXT NOT NULL,
            industry        TEXT,
            country         TEXT,
            account_manager TEXT,
            active_projects TEXT   -- comma-separated project IDs
        )
    """)
    clients = [
        ("CLI-001","Acme Corp",         "Manufacturing",    "USA",     "Henry Brown",   "PRJ-001,PRJ-006"),
        ("CLI-002","GlobalTech Inc",    "Technology",       "Germany", "Emma Rodriguez","PRJ-002"),
        ("CLI-003","HealthPlus",        "Healthcare",       "Canada",  "Carol Williams","PRJ-003"),
        ("CLI-004","RetailChain",       "Retail",           "Mexico",  "Henry Brown",   "PRJ-004"),
        ("CLI-005","EduLearn",          "Education",        "USA",     "Bob Johnson",   "PRJ-005,PRJ-007"),
        ("CLI-006","FinServe Bank",     "Finance",          "UK",      "Carol Williams","PRJ-006"),
        ("CLI-007","AgriSmart",         "Agriculture",      "Brazil",  "Emma Rodriguez","PRJ-007"),
        ("CLI-008","LogiTrack",         "Logistics",        "Spain",   "Henry Brown",   "PRJ-001"),
        ("CLI-009","MediaWave",         "Media",            "France",  "Bob Johnson",   "PRJ-002,PRJ-004"),
        ("CLI-010","CyberShield",       "Cybersecurity",    "Israel",  "Henry Brown",   "PRJ-003"),
    ]
    c.executemany("INSERT INTO clients VALUES (?,?,?,?,?,?)", clients)

    # ── TABLE: SKILL_CERTIFICATIONS ───────────────────────────────────────────
    c.execute("DROP TABLE IF EXISTS skill_certifications")
    c.execute("""
        CREATE TABLE skill_certifications (
            certification_id   TEXT PRIMARY KEY,
            employee_id        TEXT NOT NULL,
            certification_name TEXT NOT NULL,
            provider           TEXT,
            issue_date         TEXT,
            expiration_date    TEXT
        )
    """)
    certs = [
        ("CERT-001","EMP-001","AWS Solutions Architect",     "Amazon", "2023-01-15","2026-01-15"),
        ("CERT-002","EMP-001","Certified Kubernetes Admin",  "CNCF",   "2022-06-01","2025-06-01"),
        ("CERT-003","EMP-004","Docker Certified Associate",  "Docker", "2023-05-20","2026-05-20"),
        ("CERT-004","EMP-004","Google Cloud Professional",   "Google", "2024-01-10","2027-01-10"),
        ("CERT-005","EMP-006","AWS Machine Learning Spec.",  "Amazon", "2023-09-01","2026-09-01"),
        ("CERT-006","EMP-006","Microsoft Azure Data Eng.",   "Microsoft","2024-03-15","2027-03-15"),
        ("CERT-007","EMP-003","CPA - Certified Public Acc.", "AICPA",  "2019-12-01","2028-12-01"),
        ("CERT-008","EMP-007","Oracle Certified Professional","Oracle", "2022-08-10","2025-08-10"),
        ("CERT-009","EMP-009","Meta React Developer",        "Meta",   "2024-02-20","2027-02-20"),
        ("CERT-010","EMP-002","SHRM-CP HR Certification",    "SHRM",   "2021-04-05","2024-04-05"),
        ("CERT-011","EMP-010","ARM Cortex Specialist",       "ARM",    "2020-11-01","2026-11-01"),
        ("CERT-012","EMP-005","Google Analytics Certified",  "Google", "2024-05-01","2025-05-01"),
    ]
    c.executemany("INSERT INTO skill_certifications VALUES (?,?,?,?,?,?)", certs)

    conn.commit()
    conn.close()

    # Print summary
    conn2 = sqlite3.connect(DB_PATH)
    c2 = conn2.cursor()
    for tbl in ["tickets","employees","projects","clients","skill_certifications"]:
        c2.execute(f"SELECT COUNT(*) FROM {tbl}")
        count = c2.fetchone()[0]
        print(f"  {tbl}: {count} rows")
    conn2.close()
    print(f"\nDatabase created at: {DB_PATH}")


if __name__ == "__main__":
    print("Setting up database...")
    setup()
    print("Done!")
