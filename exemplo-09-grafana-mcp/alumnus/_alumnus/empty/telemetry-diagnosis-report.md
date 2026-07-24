# Comprehensive Diagnosis Report: DB Leaky Connections (500 Errors)

Date: 2026-07-24
Window analyzed: 2026-07-24T18:30:00Z to 2026-07-24T19:50:00Z
Endpoint: /students/db-leaky-connections
Service: alumnus_app_a780

---

## 1) Metrics Investigation (Prometheus)

Prometheus datasource is available, but the currently exposed Grafana MCP toolset in this environment supports metric discovery (names/metadata) and does not expose direct PromQL query execution. Because of that, quantitative HTTP 500 rates and latency distributions were computed from correlated Loki request-completion logs.

Discovered HTTP metrics in Prometheus include:

- http_server_duration_milliseconds_bucket
- http_server_duration_milliseconds_count
- http_server_duration_milliseconds_sum
- http_client_request_duration_seconds_bucket
- http_client_request_duration_seconds_count
- http_client_request_duration_seconds_sum

Computed endpoint metrics from Loki correlation (incoming request + request completed, joined by reqId):

- Incoming requests to endpoint: 100
- Completed requests to endpoint: 99
- HTTP status distribution: 99 x 500
- Average failure latency: 1001.93 ms

Interpretation:

- Failures cluster around ~1 second, strongly indicating connection acquisition timeout.

---

## 2) Logs Investigation (Loki)

Correlation query strategy:

- Query all service logs in the window.
- Isolate "incoming request" with req_url=/students/db-leaky-connections.
- Join with "request completed" on reqId.
- Extract structured error fields from "Error processing request" entries.

Key findings:

- 100 error events with the same message:
  - timeout exceeded when trying to connect
- Error burst time range:
  - First observed: 2026-07-24T19:46:40Z
  - Last observed: 2026-07-24T19:49:58Z
- Canonical stack trace:

```text
Error: timeout exceeded when trying to connect
    at /Users/mauricioaires/machine-larning/exemplo-09-grafana-mcp/alumnus/_alumnus/node_modules/pg-pool/index.js:45:11
    at async DbLeakyConnectionsScenario.createConnection (file:///Users/mauricioaires/machine-larning/exemplo-09-grafana-mcp/alumnus/_alumnus/src/scenarios/db-leaky-connections/main.ts:52:20)
    at async Object.<anonymous> (file:///Users/mauricioaires/machine-larning/exemplo-09-grafana-mcp/alumnus/_alumnus/src/scenarios/db-leaky-connections/main.ts:84:24)
```

Pattern over time:

- In this captured window, endpoint traffic is in a sustained failure phase (500-only).
- This is consistent with a previously exhausted connection pool state.

---

## 3) Traces Investigation (Tempo)

Tempo evidence for the same endpoint:

- span.http.target values include:
  - /students/db-leaky-connections
- span.http.status_code values include:
  - 200
  - 500
- status values include:
  - ok
  - error
  - unset

Relevant span names observed in this service:

- GET /students/db-leaky-connections
- request
- handler - fastify -> @fastify/otel
- pg.connect
- pg.query:SELECT alumnus_app_a780

Important negative signal:

- No span names indicating release/cleanup were found (for example, no pg.release-like span), while connection acquisition spans are present.
- This aligns with missing connection return to pool.

---

## 4) Root Cause Analysis

Root cause:

- PostgreSQL pooled connections are acquired but not reliably released.
- Once pool slots are exhausted, subsequent requests time out on pool.connect().

Exact code locations from stack traces:

- src/scenarios/db-leaky-connections/main.ts:52
- src/scenarios/db-leaky-connections/main.ts:84

Technical conclusion:

- The failing behavior matches connection leak mechanics: acquire succeeds until pool is full, then all further acquisitions timeout at ~1s.

---

## 5) Telemetry Correlation Table

| Telemetry              | Observed evidence                                                                 | Correlated conclusion                                              |
| ---------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Prometheus capability  | HTTP metrics discovered, no direct PromQL execution available in active MCP tools | Quantitative error-rate calculation performed via Loki correlation |
| Loki request lifecycle | 100 incoming, 99 completed, 99 x 500 for target endpoint                          | Endpoint currently in full failure phase                           |
| Loki latency           | Avg failure latency 1001.93 ms                                                    | Timeout threshold behavior                                         |
| Loki errors            | 100 identical errors: timeout exceeded when trying to connect                     | Pool acquisition timeout                                           |
| Loki stack trace       | main.ts:52 and main.ts:84 in db-leaky-connections scenario                        | Precise failing path identified                                    |
| Tempo spans            | GET/request/handler plus pg.connect and pg.query present                          | DB operation path is active                                        |
| Tempo status codes     | Both 200 and 500 exist for this endpoint in trace data                            | Confirms endpoint transitions between healthy and failed states    |
| Tempo negative signal  | No release/cleanup span names found                                               | Supports missing release hypothesis                                |

---

## 6) Fix

Use guaranteed release with try/finally around pool.connect():

```ts
const client = await this.pool.connect();
try {
  const result = await client.query("SELECT * FROM students LIMIT 1");
  return reply.send({ students: result.rows });
} finally {
  client.release();
}
```

---

## 7) Verification Plan

1. Call POST /students/db-leaky-connections/reset.
2. Send sequential GET requests to /students/db-leaky-connections.
3. Expected before fix:
   - first requests succeed, then repeated ~1s 500 failures.
4. Expected after fix:
   - sustained 200 responses, no pool timeout errors.
5. Re-check telemetry:
   - Loki: no timeout exceeded when trying to connect errors.
   - Tempo: normal request traces continue without growing timeout failures.

---

## Final Diagnosis

The 500 errors are caused by a DB connection leak in src/scenarios/db-leaky-connections/main.ts, where connections obtained from pool.connect() are not consistently released. This exhausts the small pool and causes deterministic ~1s acquisition timeouts, observed consistently in logs and trace status signals.
