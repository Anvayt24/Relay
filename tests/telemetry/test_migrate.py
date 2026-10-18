from relay.telemetry.migrate import discover


def test_migrations_are_found_in_order() -> None:
    migrations = discover()
    assert [m.version for m in migrations] == sorted(m.version for m in migrations)
    assert migrations[0].name == "0001_requests.sql"
    assert "CREATE TABLE requests" in migrations[0].sql


def test_request_id_is_not_the_primary_key() -> None:
    # A client-supplied ID as primary key would let one repeated X-Request-ID fail a whole
    # COPY batch. Guard the decision.
    sql = discover()[0].sql
    assert "request_id  text        NOT NULL," in sql
    assert "GENERATED ALWAYS AS IDENTITY PRIMARY KEY" in sql
