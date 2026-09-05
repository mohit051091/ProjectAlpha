"""Test DuckDB time casts."""
import duckdb
con = duckdb.connect()
tests = [
    ("SELECT ('2026-07-13 03:45:00+00:00'::TIMESTAMPTZ + INTERVAL '5 hours' + INTERVAL '30 minutes')::TIME AS t", "Add 5:30 then ::TIME"),
    ("SELECT '2026-07-13 03:45:00+00:00'::TIMESTAMPTZ::TIME AS t", "Direct ::TIME"),
    ("SELECT ('2026-07-13 03:45:00+00:00'::TIMESTAMPTZ AT TIME ZONE 'Asia/Kolkata')::TIME AS t", "AT TIME ZONE then ::TIME"),
    ("SELECT EXTRACT(HOUR FROM '2026-07-13 03:45:00+00:00'::TIMESTAMPTZ) AS h", "EXTRACT HOUR"),
    ("SELECT CAST('2026-07-13 03:45:00+00:00'::TIMESTAMPTZ AS TIME) AS t", "CAST AS TIME"),
    ("SELECT ('2026-07-13 03:45:00+00:00'::TIMESTAMPTZ::TIMESTAMP)::TIME AS t", "::TIMESTAMP then ::TIME"),
    ("SELECT CAST(ts AS TIME) FROM (SELECT '2026-07-13 03:45:00+00:00'::TIMESTAMPTZ AS ts) sub", "CAST(ts AS TIME)"),
]
for query, label in tests:
    try:
        result = con.execute(query).fetchall()
        print(f"[OK] {label}: {result}")
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
