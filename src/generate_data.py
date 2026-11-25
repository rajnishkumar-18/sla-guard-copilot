from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RUNBOOKS_DIR = DATA_DIR / "runbooks"


def generate_tickets(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    priorities = ["P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"]
    priority_probs = [0.12, 0.25, 0.43, 0.20]
    sla_map = {
        "P1 - Critical": 4,
        "P2 - High": 8,
        "P3 - Medium": 24,
        "P4 - Low": 48,
    }

    categories = [
        "Database & SQL",
        "Network & VPN",
        "Cloud Infrastructure",
        "Identity & Access (IAM)",
        "Application Error",
    ]
    teams = ["L1-ServiceDesk", "L2-InfraOps", "L3-DBA", "L2-NetOps", "SecOps"]

    p_col = rng.choice(priorities, size=n_samples, p=priority_probs)
    c_col = rng.choice(categories, size=n_samples, p=[0.22, 0.20, 0.20, 0.18, 0.20])
    t_col = rng.choice(teams, size=n_samples, p=[0.30, 0.25, 0.15, 0.15, 0.15])

    open_queue = rng.integers(3, 65, size=n_samples)
    reassignments = rng.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.42, 0.26, 0.16, 0.09, 0.05, 0.02])
    created_hour = rng.integers(0, 24, size=n_samples)
    is_weekend = rng.choice([0, 1], size=n_samples, p=[0.78, 0.22])
    sla_target = np.array([sla_map[p] for p in p_col])

    logit = -3.2
    logit = logit + np.where(p_col == "P1 - Critical", 1.8, 0.0)
    logit = logit + np.where(p_col == "P2 - High", 0.9, 0.0)
    logit = logit + np.where(c_col == "Database & SQL", 0.7, 0.0)
    logit = logit + np.where(c_col == "Network & VPN", 0.5, 0.0)
    logit = logit + np.where(t_col == "L1-ServiceDesk", 0.4, 0.0)
    logit = logit + (open_queue - 25) * 0.06
    logit = logit + reassignments * 0.65
    logit = logit + np.where((created_hour >= 21) | (created_hour <= 5), 0.75, 0.0)
    logit = logit + is_weekend * 0.6
    logit = logit + rng.normal(0, 0.45, size=n_samples)

    breach_prob = 1 / (1 + np.exp(-logit))
    sla_breached = (rng.random(n_samples) < breach_prob).astype(int)

    actual_hours = np.where(
        sla_breached == 1,
        sla_target * rng.uniform(1.05, 2.2, size=n_samples),
        sla_target * rng.uniform(0.20, 0.92, size=n_samples),
    ).round(1)

    df = pd.DataFrame(
        {
            "ticket_id": [f"INC-{10000 + i}" for i in range(n_samples)],
            "priority": p_col,
            "category": c_col,
            "assigned_group": t_col,
            "open_tickets_in_queue": open_queue,
            "reassignment_count": reassignments,
            "created_hour": created_hour,
            "is_weekend": is_weekend,
            "sla_target_hours": sla_target,
            "actual_resolution_hours": actual_hours,
            "sla_breached": sla_breached,
        }
    )
    return df


def generate_runbooks() -> None:
    RUNBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    runbooks = {
        "database_sql_sop.txt": """TITLE: Enterprise Database & SQL Latency / Deadlock Runbook
CATEGORY: Database & SQL
ESCALATION TEAM: L3-DBA (Primary), L2-InfraOps (Secondary)
SLA POLICY: P1 tickets must be acknowledged within 15 minutes and escalated to L3-DBA if unresolved after 45 minutes.

SYMPTOMS:
- Query timeouts, high DB CPU (>85%), connection pool exhaustion, transaction deadlocks, slow checkout or ERP reporting.

DIAGNOSTIC & RESOLUTION STEPS:
1. Check active blocking sessions using pg_stat_activity (PostgreSQL) or sp_who2 / sys.dm_exec_requests (SQL Server).
2. Identify long-running unindexed queries or table locks holding shared/exclusive locks beyond 60 seconds.
3. Verify application connection pool metrics (HikariCP / PgBouncer); recycle stale connections if pool utilization exceeds 90%.
4. If a deadlock loop persists, terminate the non-critical blocking PID after logging query text for RCA.
5. Trigger read-replica failover if primary node CPU stays above 90% for 10 consecutive minutes.
""",
        "network_vpn_sop.txt": """TITLE: Corporate Network, Firewall & VPN Outage Runbook
CATEGORY: Network & VPN
ESCALATION TEAM: L2-NetOps (Primary), SecOps (Firewall/Zero-Trust)
SLA POLICY: P1 regional VPN outage requires immediate bridge call with L2-NetOps and ISP vendor ticket within 20 minutes.

SYMPTOMS:
- Remote employees unable to connect to Cisco AnyConnect / GlobalProtect VPN, packet drops, DNS resolution failure, SSL handshake timeout.

DIAGNOSTIC & RESOLUTION STEPS:
1. Check VPN gateway health dashboard and verify active concurrent tunnel count against license ceiling.
2. Run traceroute and ping tests from internal jump host to edge firewall and ISP gateway to isolate packet loss.
3. Verify RADIUS / Azure AD MFA authentication logs to confirm whether failures are credential/token related rather than network transport.
4. Shift traffic to Secondary DR VPN Concentrator if primary gateway latency exceeds 150ms or drops >5% packets.
5. Broadcast status update on IT Ops Slack/Teams channel with ETA and workaround gateway URL.
""",
        "cloud_infra_sop.txt": """TITLE: Cloud Infrastructure, Kubernetes & Server Spike Runbook
CATEGORY: Cloud Infrastructure
ESCALATION TEAM: L2-InfraOps (Primary), Cloud SRE On-Call
SLA POLICY: Auto-scaling failure or production VM crash must escalate to L2-InfraOps within 30 minutes.

SYMPTOMS:
- Azure VM / AWS EC2 high CPU or memory OOMKill, Kubernetes CrashLoopBackOff, disk space >95%, Load Balancer 502/503 errors.

DIAGNOSTIC & RESOLUTION STEPS:
1. Inspect node metrics (CPU, Memory, Disk IOPS) in Azure Monitor / CloudWatch to distinguish traffic surge vs memory leak.
2. For Kubernetes pods in CrashLoopBackOff, run kubectl describe pod and kubectl logs --previous to capture stack trace.
3. Clear rotated application logs or expand persistent volume if root/var partition utilization exceeds 90%.
4. Verify Horizontal Pod Autoscaler (HPA) or VM Scale Set quotas; manually scale replica count by +2 to absorb immediate load.
5. Roll back the most recent deployment revision if errors started within 30 minutes of a CI/CD release.
""",
        "iam_access_sop.txt": """TITLE: Identity, SSO, Active Directory & Access Management Runbook
CATEGORY: Identity & Access (IAM)
ESCALATION TEAM: SecOps (Primary), L1-ServiceDesk (Password/MFA Reset)
SLA POLICY: Privileged access lockouts or SSO outages affecting >20 users are P1 and escalate directly to SecOps.

SYMPTOMS:
- Azure AD / Okta SSO SAML assertion failure, repeated account lockouts, MFA push not received, expired service principal secrets.

DIAGNOSTIC & RESOLUTION STEPS:
1. Inspect Azure AD Sign-in Logs / Okta System Log for error codes (e.g., AADSTS50076 MFA required, AADSTS7000215 expired client secret).
2. Confirm whether lockout is isolated to one user (password spray / cached mobile credentials) or tenant-wide (expired SAML certificate).
3. For expired service principal or API certificate, rotate key in Azure Key Vault and restart dependent app services.
4. Never bypass MFA policies; use temporary access pass (TAP) with manager approval for verified emergency lockouts.
5. Document ticket ID and approval trail in IAM audit log for SOC2/ISO compliance.
""",
        "application_error_sop.txt": """TITLE: Enterprise Web & Microservice Application Error Runbook
CATEGORY: Application Error
ESCALATION TEAM: L2-InfraOps (First Triage), App Engineering On-Call
SLA POLICY: Production 5xx error rate >2% requires P1 escalation and rollback evaluation within 30 minutes.

SYMPTOMS:
- HTTP 500/502/504 errors, downstream REST API timeouts, message queue backlog (Kafka/RabbitMQ), frontend blank screen.

DIAGNOSTIC & RESOLUTION STEPS:
1. Check APM traces (Application Insights / Datadog) to pinpoint the failing microservice endpoint and latency percentile (p95/p99).
2. Correlate timestamp with recent deployments, feature flag flips, or upstream third-party API outages.
3. Inspect dead-letter queues (DLQ) and consumer lag; restart stalled worker pods if downstream service is healthy.
4. Enable circuit breaker or fallback cache for non-critical downstream integrations causing thread pool starvation.
5. Capture correlation IDs and stack traces before executing rolling restart or release rollback.
""",
    }

    for filename, content in runbooks.items():
        (RUNBOOKS_DIR / filename).write_text(content.strip(), encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_tickets(n_samples=5000)
    out_csv = DATA_DIR / "raw_tickets.csv"
    df.to_csv(out_csv, index=False)
    generate_runbooks()
    print(f"Generated {len(df)} tickets at {out_csv}")
    print(f"SLA Breach Rate: {df['sla_breached'].mean() * 100:.2f}%")
    print(f"Generated 5 enterprise SOP runbooks in {RUNBOOKS_DIR}")


if __name__ == "__main__":
    main()
