"""Deterministic generator script for the synthetic enterprise reference corpus.

Generates at least 120 interconnected documents representing a fictional enterprise
technology company ('Aegis Global Technologies') with microservices, teams, APIs,
databases, infrastructure, security CVEs, and incident reports.

Usage:
    python -m src.data.generate_corpus
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_corpus_documents() -> list[dict[str, Any]]:
    """Build a deterministic list of 120+ structured enterprise documents."""
    docs: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. CORE TEAMS (10 Documents)
    # -------------------------------------------------------------------------
    teams_data = [
        {
            "id": "doc-team-checkout-engineering",
            "entity_id": "team:checkout-engineering",
            "name": "Checkout Engineering Team",
            "aliases": ["Checkout Team", "checkout-eng", "Core Checkout"],
            "lead": "Sarah Jenkins",
            "slack": "#proj-checkout-eng",
            "charter": "Responsible for customer checkout flows, order processing, and payment gateway orchestration.",
            "services": [
                "service:checkout-service",
                "service:payment-gateway",
                "service:cart-service",
            ],
            "repos": ["repo:checkout-service-repo", "repo:payment-gateway-repo"],
        },
        {
            "id": "doc-team-identity-access",
            "entity_id": "team:identity-access",
            "name": "Identity and Access Management Team",
            "aliases": ["IAM Team", "auth-team", "Identity Team"],
            "lead": "Marcus Vance",
            "slack": "#proj-iam-auth",
            "charter": "Maintains single sign-on, authentication, OAuth2 token issuance, and authorization policies across all platforms.",
            "services": ["service:auth-service", "service:user-directory-service"],
            "repos": ["repo:auth-service-repo"],
        },
        {
            "id": "doc-team-data-platform",
            "entity_id": "team:data-platform",
            "name": "Data Platform Team",
            "aliases": ["Data Infra", "data-platform-eng"],
            "lead": "Elena Rostova",
            "slack": "#proj-data-platform",
            "charter": "Operates managed PostgreSQL clusters, Redis caches, Apache Kafka event streaming, and analytics infrastructure.",
            "services": ["service:data-pipeline-service", "service:analytics-aggregator"],
            "repos": ["repo:data-pipeline-repo"],
        },
        {
            "id": "doc-team-notification-messaging",
            "entity_id": "team:notification-messaging",
            "name": "Notification and Messaging Team",
            "aliases": ["Notification Team", "comms-team", "messaging-eng"],
            "lead": "Devon Miller",
            "slack": "#proj-comms-eng",
            "charter": "Manages transactional email, SMS, push notifications, and customer alert delivery systems.",
            "services": ["service:notification-service", "service:sms-relay-service"],
            "repos": ["repo:notification-service-repo"],
        },
        {
            "id": "doc-team-inventory-fulfillment",
            "entity_id": "team:inventory-fulfillment",
            "name": "Inventory and Fulfillment Team",
            "aliases": ["Warehouse Team", "inventory-eng", "Logistics Team"],
            "lead": "Kenji Sato",
            "slack": "#proj-fulfillment-eng",
            "charter": "Oversees real-time warehouse inventory reservations, stock tracking, and supplier logistics integration.",
            "services": ["service:inventory-service", "service:warehouse-service"],
            "repos": ["repo:inventory-service-repo"],
        },
        {
            "id": "doc-team-catalog-search",
            "entity_id": "team:catalog-search",
            "name": "Catalog and Search Team",
            "aliases": ["Search Team", "catalog-eng", "Discovery Team"],
            "lead": "Priya Sharma",
            "slack": "#proj-catalog-search",
            "charter": "Builds and optimizes product catalog data schemas, full-text Elasticsearch clusters, and recommendation ranking.",
            "services": ["service:catalog-service", "service:search-indexer"],
            "repos": ["repo:catalog-service-repo"],
        },
        {
            "id": "doc-team-security-secops",
            "entity_id": "team:security-secops",
            "name": "Enterprise Security and SecOps Team",
            "aliases": ["Security Team", "InfoSec", "SecOps"],
            "lead": "Rachel Adams",
            "slack": "#proj-security-ops",
            "charter": "Conducts vulnerability scanning, threat assessments, CVE triage, network policy governance, and incident response.",
            "services": ["service:audit-logger-service", "service:waf-manager"],
            "repos": ["repo:security-tools-repo"],
        },
        {
            "id": "doc-team-customer-support-eng",
            "entity_id": "team:customer-support-eng",
            "name": "Customer Support Engineering Team",
            "aliases": ["Support Eng", "Care Tech"],
            "lead": "Carlos Mendez",
            "slack": "#proj-support-tech",
            "charter": "Develops internal CRM tools, dispute resolution portals, and customer ticketing interfaces.",
            "services": ["service:ticketing-service", "service:crm-sync-service"],
            "repos": ["repo:ticketing-service-repo"],
        },
        {
            "id": "doc-team-billing-finance",
            "entity_id": "team:billing-finance",
            "name": "Billing and Finance Tech Team",
            "aliases": ["Finance Tech", "billing-eng", "Ledger Team"],
            "lead": "Ananya Patel",
            "slack": "#proj-billing-fin",
            "charter": "Maintains double-entry general ledgers, recurring subscription billing, invoicing, and tax calculation engines.",
            "services": ["service:billing-service", "service:tax-calculator-service"],
            "repos": ["repo:billing-service-repo"],
        },
        {
            "id": "doc-team-cloud-infrastructure",
            "entity_id": "team:cloud-infrastructure",
            "name": "Cloud Infrastructure and SRE Team",
            "aliases": ["Infra Team", "SRE", "DevOps", "Platform SRE"],
            "lead": "Liam O'Connor",
            "slack": "#proj-cloud-infra",
            "charter": "Provisions Kubernetes clusters, AWS VPCs, egress gateways, and observability monitoring stacks.",
            "services": ["service:cluster-orchestrator", "service:monitoring-collector"],
            "repos": ["repo:infra-terraform-repo"],
        },
    ]

    for t in teams_data:
        docs.append(
            {
                "document_id": t["id"],
                "source_uri": f"docs/teams/{t['id']}.md",
                "title": f"Team Charter: {t['name']}",
                "doc_type": "team_charter",
                "content": (
                    f"# Team Charter: {t['name']}\n\n"
                    f"**Team Identifier:** `{t['entity_id']}`\n"
                    f"**Team Lead:** {t['lead']}\n"
                    f"**Communication Channel:** {t['slack']}\n"
                    f"**Known Aliases:** {', '.join(t['aliases'])}\n\n"
                    f"## Charter & Responsibilities\n"
                    f"{t['charter']}\n\n"
                    f"## Owned Services & Artifacts\n"
                    f"The {t['name']} is the sole designated owner (`OWNED_BY`) for the following systems:\n"
                    + "\n".join([f"- Service: `{s}`" for s in t["services"]])
                    + "\n\nRepositories managed:\n"
                    + "\n".join([f"- Repository: `{r}`" for r in t["repos"]])
                ),
                "metadata": {
                    "entity_id": t["entity_id"],
                    "entity_type": "Team",
                    "aliases": t["aliases"],
                    "lead": t["lead"],
                },
            }
        )

    # -------------------------------------------------------------------------
    # 2. CORE SERVICES (20 Documents)
    # -------------------------------------------------------------------------
    services_data = [
        {
            "id": "doc-service-checkout-service",
            "entity_id": "service:checkout-service",
            "name": "Checkout Service",
            "aliases": ["checkout-svc", "Checkout Core"],
            "team": "team:checkout-engineering",
            "repo": "repo:checkout-service-repo",
            "apis_exposed": ["api:create-order-v2", "api:get-order-status-v1"],
            "apis_called": [
                "api:process-charge-v2",
                "api:reserve-inventory-v1",
                "api:validate-token-v1",
            ],
            "depends_on": [
                "service:payment-gateway",
                "service:inventory-service",
                "service:auth-service",
            ],
            "reads_from": ["database:customer-orders-pg"],
            "writes_to": ["database:customer-orders-pg"],
            "infra": ["infra:k8s-prod-cluster-us-east", "infra:kafka-event-bus"],
            "desc": "Orchestrates order placement, inventory validation, and payment execution during user checkout.",
        },
        {
            "id": "doc-service-payment-gateway",
            "entity_id": "service:payment-gateway",
            "name": "Payment Gateway Service",
            "aliases": ["pay-gw", "payment-service", "Payment Svc"],
            "team": "team:checkout-engineering",
            "repo": "repo:payment-gateway-repo",
            "apis_exposed": ["api:process-charge-v2", "api:refund-charge-v1"],
            "apis_called": ["api:validate-token-v1", "api:record-ledger-entry-v1"],
            "depends_on": ["service:auth-service", "service:billing-service"],
            "reads_from": ["database:payment-vault-pg"],
            "writes_to": ["database:payment-vault-pg"],
            "infra": ["infra:k8s-prod-cluster-us-east", "infra:pci-isolated-vpc"],
            "desc": "Processes credit card, ACH, and digital wallet transactions with strict PCI-DSS compliance.",
        },
        {
            "id": "doc-service-auth-service",
            "entity_id": "service:auth-service",
            "name": "Authentication Service",
            "aliases": ["auth-svc", "IAM Core", "OAuth Service"],
            "team": "team:identity-access",
            "repo": "repo:auth-service-repo",
            "apis_exposed": ["api:validate-token-v1", "api:issue-jwt-v2"],
            "apis_called": [],
            "depends_on": ["service:user-directory-service"],
            "reads_from": ["database:auth-session-redis", "database:user-directory-pg"],
            "writes_to": ["database:auth-session-redis"],
            "infra": ["infra:k8s-prod-cluster-us-east"],
            "desc": "Issues cryptographically signed JWT tokens and validates session tickets across all enterprise APIs.",
        },
        {
            "id": "doc-service-inventory-service",
            "entity_id": "service:inventory-service",
            "name": "Inventory Reservation Service",
            "aliases": ["inventory-svc", "Stock Reservation Service"],
            "team": "team:inventory-fulfillment",
            "repo": "repo:inventory-service-repo",
            "apis_exposed": ["api:reserve-inventory-v1", "api:check-stock-v1"],
            "apis_called": ["api:notify-warehouse-v1"],
            "depends_on": ["service:warehouse-service"],
            "reads_from": ["database:inventory-pg"],
            "writes_to": ["database:inventory-pg"],
            "infra": ["infra:k8s-prod-cluster-us-east", "infra:kafka-event-bus"],
            "desc": "Maintains atomic inventory reservations preventing overselling across regional fulfillment centers.",
        },
        {
            "id": "doc-service-catalog-service",
            "entity_id": "service:catalog-service",
            "name": "Product Catalog Service",
            "aliases": ["catalog-svc", "Product Registry"],
            "team": "team:catalog-search",
            "repo": "repo:catalog-service-repo",
            "apis_exposed": ["api:get-product-details-v1", "api:search-products-v2"],
            "apis_called": [],
            "depends_on": [],
            "reads_from": ["database:catalog-mongodb", "database:search-elasticsearch"],
            "writes_to": ["database:catalog-mongodb"],
            "infra": ["infra:k8s-prod-cluster-us-east"],
            "desc": "Serves product metadata, category hierarchies, pricing, and high-speed search queries.",
        },
        {
            "id": "doc-service-notification-service",
            "entity_id": "service:notification-service",
            "name": "Notification Dispatch Service",
            "aliases": ["notification-svc", "Comms Dispatcher"],
            "team": "team:notification-messaging",
            "repo": "repo:notification-service-repo",
            "apis_exposed": ["api:send-email-v1", "api:send-push-v1"],
            "apis_called": ["api:send-sms-v1"],
            "depends_on": ["service:sms-relay-service"],
            "reads_from": ["database:notification-pg"],
            "writes_to": ["database:notification-pg"],
            "infra": ["infra:k8s-prod-cluster-us-east", "infra:kafka-event-bus"],
            "desc": "Dispatches transactional emails, push notifications, and consumer alerts triggered by platform events.",
        },
        {
            "id": "doc-service-billing-service",
            "entity_id": "service:billing-service",
            "name": "Billing and Invoicing Service",
            "aliases": ["billing-svc", "Ledger Engine"],
            "team": "team:billing-finance",
            "repo": "repo:billing-service-repo",
            "apis_exposed": ["api:record-ledger-entry-v1", "api:generate-invoice-v2"],
            "apis_called": ["api:calculate-tax-v1"],
            "depends_on": ["service:tax-calculator-service"],
            "reads_from": ["database:financial-ledger-pg"],
            "writes_to": ["database:financial-ledger-pg"],
            "infra": ["infra:k8s-prod-cluster-us-east"],
            "desc": "Maintains immutable double-entry ledger records and computes recurring subscription invoicing.",
        },
        {
            "id": "doc-service-warehouse-service",
            "entity_id": "service:warehouse-service",
            "name": "Warehouse Management Service",
            "aliases": ["warehouse-svc", "WMS Core"],
            "team": "team:inventory-fulfillment",
            "repo": "repo:inventory-service-repo",
            "apis_exposed": ["api:notify-warehouse-v1"],
            "apis_called": [],
            "depends_on": [],
            "reads_from": ["database:inventory-pg"],
            "writes_to": ["database:inventory-pg"],
            "infra": ["infra:k8s-prod-cluster-us-west"],
            "desc": "Coordinates automated robotic picking queues and packing slips across logistics warehouses.",
        },
        {
            "id": "doc-service-sms-relay-service",
            "entity_id": "service:sms-relay-service",
            "name": "SMS Relay Gateway",
            "aliases": ["sms-relay", "Twilio Gateway"],
            "team": "team:notification-messaging",
            "repo": "repo:notification-service-repo",
            "apis_exposed": ["api:send-sms-v1"],
            "apis_called": [],
            "depends_on": [],
            "reads_from": [],
            "writes_to": [],
            "infra": ["infra:k8s-prod-cluster-us-east"],
            "desc": "Integrates third-party carrier gateways to deliver high-priority 2FA tokens and SMS alerts.",
        },
        {
            "id": "doc-service-tax-calculator-service",
            "entity_id": "service:tax-calculator-service",
            "name": "Tax Calculation Service",
            "aliases": ["tax-svc", "Avalara Adapter"],
            "team": "team:billing-finance",
            "repo": "repo:billing-service-repo",
            "apis_exposed": ["api:calculate-tax-v1"],
            "apis_called": [],
            "depends_on": [],
            "reads_from": [],
            "writes_to": [],
            "infra": ["infra:k8s-prod-cluster-us-east"],
            "desc": "Determines state, municipal, and VAT tax liability based on buyer shipping location.",
        },
        {
            "id": "doc-service-user-directory-service",
            "entity_id": "service:user-directory-service",
            "name": "User Directory Service",
            "aliases": ["user-dir", "Account Registry"],
            "team": "team:identity-access",
            "repo": "repo:auth-service-repo",
            "apis_exposed": ["api:get-user-profile-v1"],
            "apis_called": [],
            "depends_on": [],
            "reads_from": ["database:user-directory-pg"],
            "writes_to": ["database:user-directory-pg"],
            "infra": ["infra:k8s-prod-cluster-us-east"],
            "desc": "Stores user profiles, credentials, hashing salts, and account security settings.",
        },
        {
            "id": "doc-service-cart-service",
            "entity_id": "service:cart-service",
            "name": "Shopping Cart Service",
            "aliases": ["cart-svc", "Basket Engine"],
            "team": "team:checkout-engineering",
            "repo": "repo:checkout-service-repo",
            "apis_exposed": ["api:add-cart-item-v1"],
            "apis_called": ["api:check-stock-v1"],
            "depends_on": ["service:inventory-service"],
            "reads_from": ["database:cart-redis"],
            "writes_to": ["database:cart-redis"],
            "infra": ["infra:k8s-prod-cluster-us-east"],
            "desc": "Provides low-latency temporary storage of active buyer shopping baskets and promo codes.",
        },
        {
            "id": "doc-service-search-indexer",
            "entity_id": "service:search-indexer",
            "name": "Search Pipeline Indexer",
            "aliases": ["search-indexer", "Elastic Ingestion"],
            "team": "team:catalog-search",
            "repo": "repo:catalog-service-repo",
            "apis_exposed": [],
            "apis_called": ["api:get-product-details-v1"],
            "depends_on": ["service:catalog-service"],
            "reads_from": ["database:catalog-mongodb"],
            "writes_to": ["database:search-elasticsearch"],
            "infra": ["infra:k8s-prod-cluster-us-east", "infra:kafka-event-bus"],
            "desc": "Consumes product update events and indexes documents into Elasticsearch for fast search.",
        },
        {
            "id": "doc-service-data-pipeline-service",
            "entity_id": "service:data-pipeline-service",
            "name": "Data Pipeline Ingestion Engine",
            "aliases": ["data-pipeline", "ETL Engine"],
            "team": "team:data-platform",
            "repo": "repo:data-pipeline-repo",
            "apis_exposed": [],
            "apis_called": [],
            "depends_on": [],
            "reads_from": ["database:customer-orders-pg", "database:financial-ledger-pg"],
            "writes_to": ["database:analytics-clickhouse"],
            "infra": ["infra:k8s-prod-cluster-us-east", "infra:kafka-event-bus"],
            "desc": "Extracts transactional database changelogs via Debezium and streams to ClickHouse data warehouse.",
        },
        {
            "id": "doc-service-analytics-aggregator",
            "entity_id": "service:analytics-aggregator",
            "name": "Analytics Aggregator Service",
            "aliases": ["analytics-aggregator", "BI Metric Engine"],
            "team": "team:data-platform",
            "repo": "repo:data-pipeline-repo",
            "apis_exposed": ["api:get-sales-metrics-v1"],
            "apis_called": [],
            "depends_on": [],
            "reads_from": ["database:analytics-clickhouse"],
            "writes_to": [],
            "infra": ["infra:k8s-prod-cluster-us-east"],
            "desc": "Generates aggregated business intelligence dashboards and executive sales metrics.",
        },
        {
            "id": "doc-service-ticketing-service",
            "entity_id": "service:ticketing-service",
            "name": "Support Ticketing Service",
            "aliases": ["ticketing-svc", "Zendesk Sync"],
            "team": "team:customer-support-eng",
            "repo": "repo:ticketing-service-repo",
            "apis_exposed": ["api:create-ticket-v1"],
            "apis_called": ["api:get-order-status-v1", "api:get-user-profile-v1"],
            "depends_on": ["service:checkout-service", "service:user-directory-service"],
            "reads_from": ["database:crm-ticketing-pg"],
            "writes_to": ["database:crm-ticketing-pg"],
            "infra": ["infra:k8s-prod-cluster-us-east"],
            "desc": "Tracks customer complaints, refunds, and support inquiries with live order history links.",
        },
        {
            "id": "doc-service-crm-sync-service",
            "entity_id": "service:crm-sync-service",
            "name": "CRM Sync Service",
            "aliases": ["crm-sync", "HubSpot Connector"],
            "team": "team:customer-support-eng",
            "repo": "repo:ticketing-service-repo",
            "apis_exposed": [],
            "apis_called": ["api:get-user-profile-v1"],
            "depends_on": ["service:user-directory-service"],
            "reads_from": ["database:crm-ticketing-pg"],
            "writes_to": ["database:crm-ticketing-pg"],
            "infra": ["infra:k8s-prod-cluster-us-east"],
            "desc": "Synchronizes account contact information with external enterprise CRM software.",
        },
        {
            "id": "doc-service-audit-logger-service",
            "entity_id": "service:audit-logger-service",
            "name": "Security Audit Logger",
            "aliases": ["audit-logger", "Compliance Scribe"],
            "team": "team:security-secops",
            "repo": "repo:security-tools-repo",
            "apis_exposed": ["api:log-audit-event-v1"],
            "apis_called": [],
            "depends_on": [],
            "reads_from": [],
            "writes_to": ["database:audit-logs-s3"],
            "infra": ["infra:k8s-prod-cluster-us-east", "infra:kafka-event-bus"],
            "desc": "Collects immutable tamper-evident security audit trails for SOC2 and ISO27001 compliance.",
        },
        {
            "id": "doc-service-waf-manager",
            "entity_id": "service:waf-manager",
            "name": "WAF & Rate Limiter Gateway",
            "aliases": ["waf-manager", "Edge Security Gateway"],
            "team": "team:security-secops",
            "repo": "repo:security-tools-repo",
            "apis_exposed": [],
            "apis_called": [],
            "depends_on": [],
            "reads_from": ["database:auth-session-redis"],
            "writes_to": ["database:auth-session-redis"],
            "infra": ["infra:k8s-prod-cluster-us-east", "infra:aws-edge-cloudfront"],
            "desc": "Enforces edge IP filtering, bot detection, and token-bucket API rate limiting.",
        },
        {
            "id": "doc-service-cluster-orchestrator",
            "entity_id": "service:cluster-orchestrator",
            "name": "Cluster Orchestrator Service",
            "aliases": ["cluster-orchestrator", "K8s Autoscaler"],
            "team": "team:cloud-infrastructure",
            "repo": "repo:infra-terraform-repo",
            "apis_exposed": [],
            "apis_called": [],
            "depends_on": [],
            "reads_from": [],
            "writes_to": [],
            "infra": ["infra:k8s-prod-cluster-us-east", "infra:k8s-prod-cluster-us-west"],
            "desc": "Coordinates horizontal pod autoscaling and node group provisioning across cloud regions.",
        },
    ]

    for s in services_data:
        content_lines: list[str] = [
            f"# Service Specification: {s['name']}",
            "",
            f"**Canonical Identifier:** `{s['entity_id']}`",
            f"**Owning Team (`OWNED_BY`):** `{s['team']}`",
            f"**Source Repository (`STORES_CODE`):** `{s['repo']}`",
            f"**Known Aliases:** {', '.join(s['aliases'])}",
            "",
            "## Architecture Overview",
            str(s["desc"]),
            "",
            "## Exposed APIs (`EXPOSES_API`)",
        ]
        if s["apis_exposed"]:
            content_lines.extend([f"- `{api}`" for api in s["apis_exposed"]])
        else:
            content_lines.append("- None (Internal daemon or consumer service)")

        content_lines.extend(["", "## Downstream API Calls (`CALLS_API`)"])
        if s["apis_called"]:
            content_lines.extend([f"- Invokes API `{api}`" for api in s["apis_called"]])
        else:
            content_lines.append("- Makes no downstream synchronous API calls")

        content_lines.extend(["", "## Service Dependencies (`DEPENDS_ON`)"])
        if s["depends_on"]:
            content_lines.extend([f"- Depends on `{dep}`" for dep in s["depends_on"]])
        else:
            content_lines.append("- No direct service dependencies")

        content_lines.extend(["", "## Database Storage (`READS_FROM` / `WRITES_TO`)"])
        if s["reads_from"]:
            content_lines.extend([f"- Reads from database: `{db}`" for db in s["reads_from"]])
        if s["writes_to"]:
            content_lines.extend([f"- Writes transactions to: `{db}`" for db in s["writes_to"]])
        if not s["reads_from"] and not s["writes_to"]:
            content_lines.append("- Stateless service (no direct database connections)")

        content_lines.extend(["", "## Infrastructure Deployment (`DEPLOYED_ON`)"])
        content_lines.extend([f"- Deployed on `{inf}`" for inf in s["infra"]])

        docs.append(
            {
                "document_id": s["id"],
                "source_uri": f"docs/services/{s['id']}.md",
                "title": f"Service Specification: {s['name']}",
                "doc_type": "service_spec",
                "content": "\n".join(content_lines),
                "metadata": {
                    "entity_id": s["entity_id"],
                    "entity_type": "Service",
                    "team": s["team"],
                    "repo": s["repo"],
                    "aliases": s["aliases"],
                },
            }
        )

    # -------------------------------------------------------------------------
    # 3. API CONTRACTS (15 Documents)
    # -------------------------------------------------------------------------
    apis_data = [
        (
            "api:create-order-v2",
            "Create Order API v2",
            "service:checkout-service",
            "POST /v2/orders",
            "Creates an order and initiates checkout transaction.",
            ["checkout-api", "/v2/orders"],
        ),
        (
            "api:get-order-status-v1",
            "Get Order Status API v1",
            "service:checkout-service",
            "GET /v1/orders/{id}",
            "Retrieves real-time status of placed order.",
            ["order-status-api"],
        ),
        (
            "api:process-charge-v2",
            "Process Payment Charge API v2",
            "service:payment-gateway",
            "POST /v2/charges",
            "Authorizes and captures credit card payment.",
            ["charge-api", "/v2/charges"],
        ),
        (
            "api:refund-charge-v1",
            "Refund Payment API v1",
            "service:payment-gateway",
            "POST /v1/refunds",
            "Processes refund against prior charge transaction.",
            ["refund-api"],
        ),
        (
            "api:validate-token-v1",
            "Validate OAuth2 Token API v1",
            "service:auth-service",
            "POST /v1/oauth/validate",
            "Validates JWT tokens and extracts user claims.",
            ["token-validator", "validate-jwt"],
        ),
        (
            "api:issue-jwt-v2",
            "Issue Token API v2",
            "service:auth-service",
            "POST /v2/oauth/token",
            "Authenticates credentials and returns JWT bearer token.",
            ["jwt-issuer"],
        ),
        (
            "api:reserve-inventory-v1",
            "Reserve Inventory Stock API v1",
            "service:inventory-service",
            "POST /v1/inventory/reserve",
            "Locks warehouse stock quantities for checkout.",
            ["reserve-stock-api"],
        ),
        (
            "api:check-stock-v1",
            "Check Available Stock API v1",
            "service:inventory-service",
            "GET /v1/inventory/stock/{sku}",
            "Returns current available item stock.",
            ["check-stock-api"],
        ),
        (
            "api:get-product-details-v1",
            "Get Product Details API v1",
            "service:catalog-service",
            "GET /v1/products/{id}",
            "Returns comprehensive product specifications.",
            ["product-details-api"],
        ),
        (
            "api:search-products-v2",
            "Search Catalog API v2",
            "service:catalog-service",
            "GET /v2/search",
            "Executes semantic and keyword product searches.",
            ["catalog-search-api"],
        ),
        (
            "api:send-email-v1",
            "Send Email API v1",
            "service:notification-service",
            "POST /v1/notifications/email",
            "Queues HTML email delivery via SendGrid.",
            ["send-email-api"],
        ),
        (
            "api:send-push-v1",
            "Send Push Notification API v1",
            "service:notification-service",
            "POST /v1/notifications/push",
            "Delivers mobile notifications via APNs and FCM.",
            ["push-notification-api"],
        ),
        (
            "api:send-sms-v1",
            "Send SMS API v1",
            "service:sms-relay-service",
            "POST /v1/sms/send",
            "Dispatches carrier SMS text messages.",
            ["sms-api"],
        ),
        (
            "api:record-ledger-entry-v1",
            "Record Ledger Transaction API v1",
            "service:billing-service",
            "POST /v1/ledger/entry",
            "Records double-entry financial debit/credit.",
            ["ledger-entry-api"],
        ),
        (
            "api:calculate-tax-v1",
            "Calculate Sales Tax API v1",
            "service:tax-calculator-service",
            "POST /v1/tax/calculate",
            "Calculates applicable tax rate and totals.",
            ["tax-calc-api"],
        ),
    ]

    for api_id, api_name, provider_svc, endpoint, desc, aliases in apis_data:
        doc_id = f"doc-{api_id.replace(':', '-')}"
        docs.append(
            {
                "document_id": doc_id,
                "source_uri": f"docs/apis/{doc_id}.md",
                "title": f"API Contract: {api_name}",
                "doc_type": "api_contract",
                "content": (
                    f"# API Contract: {api_name}\n\n"
                    f"**Canonical Identifier:** `{api_id}`\n"
                    f"**Exposed By (`EXPOSES_API`):** `{provider_svc}`\n"
                    f"**Endpoint:** `{endpoint}`\n"
                    f"**Known Aliases:** {', '.join(aliases)}\n\n"
                    f"## Specification & Behavior\n"
                    f"{desc}\n\n"
                    f"## Security Requirements\n"
                    f"Requires valid Bearer token validated against `service:auth-service` via `api:validate-token-v1`."
                ),
                "metadata": {
                    "entity_id": api_id,
                    "entity_type": "API",
                    "provider_service": provider_svc,
                    "endpoint": endpoint,
                    "aliases": aliases,
                },
            }
        )

    # -------------------------------------------------------------------------
    # 4. DATABASES & PERSISTENCE (10 Documents)
    # -------------------------------------------------------------------------
    dbs_data = [
        (
            "database:customer-orders-pg",
            "Customer Orders PostgreSQL",
            "PostgreSQL 16",
            "Orders, line items, and fulfillment receipts",
            "team:data-platform",
            "infra:aws-rds-postgres-cluster",
            ["PostgreSQL Orders", "orders-db"],
        ),
        (
            "database:payment-vault-pg",
            "Payment Vault PostgreSQL",
            "PostgreSQL 16 (PCI-DSS Encrypted)",
            "Encrypted customer tokens, payment methods, transaction audits",
            "team:checkout-engineering",
            "infra:pci-isolated-vpc",
            ["Payment DB", "vault-pg"],
        ),
        (
            "database:auth-session-redis",
            "Auth Session Redis Cache",
            "Redis 7 Cluster",
            "Active user session tokens, rate limit counters, OAuth tickets",
            "team:identity-access",
            "infra:aws-elasticache-redis",
            ["Redis Auth", "session-store"],
        ),
        (
            "database:user-directory-pg",
            "User Directory PostgreSQL",
            "PostgreSQL 16",
            "User accounts, password hashes, multi-factor keys",
            "team:identity-access",
            "infra:aws-rds-postgres-cluster",
            ["User DB", "identity-pg"],
        ),
        (
            "database:inventory-pg",
            "Warehouse Inventory PostgreSQL",
            "PostgreSQL 16",
            "SKU stock quantities, reservation locks, warehouse aisle mapping",
            "team:data-platform",
            "infra:aws-rds-postgres-cluster",
            ["Inventory DB", "stock-pg"],
        ),
        (
            "database:catalog-mongodb",
            "Product Catalog MongoDB",
            "MongoDB 6.0",
            "Unstructured product attribute documents, rich media specs",
            "team:catalog-search",
            "infra:mongodb-atlas-cluster",
            ["Catalog DB", "mongo-products"],
        ),
        (
            "database:search-elasticsearch",
            "Search Elasticsearch Cluster",
            "Elasticsearch 8.11",
            "Full-text tokenized product search index and lexical filters",
            "team:catalog-search",
            "infra:aws-opensearch-cluster",
            ["Elasticsearch Search", "search-cluster"],
        ),
        (
            "database:financial-ledger-pg",
            "Financial General Ledger PostgreSQL",
            "PostgreSQL 16",
            "Double-entry journal records, revenue reconciliation, invoice states",
            "team:billing-finance",
            "infra:aws-rds-postgres-cluster",
            ["Ledger DB", "finance-pg"],
        ),
        (
            "database:analytics-clickhouse",
            "Analytics ClickHouse Warehouse",
            "ClickHouse 24.3",
            "High-throughput columnar event analytics and business metrics",
            "team:data-platform",
            "infra:clickhouse-analytics-cluster",
            ["ClickHouse Analytics", "bi-warehouse"],
        ),
        (
            "database:crm-ticketing-pg",
            "Support CRM PostgreSQL",
            "PostgreSQL 16",
            "Support ticket records, resolution logs, customer interaction threads",
            "team:customer-support-eng",
            "infra:aws-rds-postgres-cluster",
            ["CRM DB", "support-pg"],
        ),
    ]

    for db_id, db_name, engine, desc, owner_team, infra_loc, aliases in dbs_data:
        doc_id = f"doc-{db_id.replace(':', '-')}"
        docs.append(
            {
                "document_id": doc_id,
                "source_uri": f"docs/databases/{doc_id}.md",
                "title": f"Database Infrastructure Spec: {db_name}",
                "doc_type": "database_spec",
                "content": (
                    f"# Database Infrastructure Spec: {db_name}\n\n"
                    f"**Canonical Identifier:** `{db_id}`\n"
                    f"**Database Engine:** {engine}\n"
                    f"**Operating Team (`OWNED_BY`):** `{owner_team}`\n"
                    f"**Hosted Infrastructure (`DEPLOYED_ON`):** `{infra_loc}`\n"
                    f"**Known Aliases:** {', '.join(aliases)}\n\n"
                    f"## Stored Data Schema & Description\n"
                    f"{desc}\n\n"
                    f"## Backup and SLA Policies\n"
                    f"Continuous WAL archiving enabled with Point-In-Time-Recovery (PITR) up to 35 days."
                ),
                "metadata": {
                    "entity_id": db_id,
                    "entity_type": "Database",
                    "engine": engine,
                    "owner_team": owner_team,
                    "infrastructure": infra_loc,
                    "aliases": aliases,
                },
            }
        )

    # -------------------------------------------------------------------------
    # 5. INFRASTRUCTURE & PLATFORM (10 Documents)
    # -------------------------------------------------------------------------
    infra_data = [
        (
            "infra:k8s-prod-cluster-us-east",
            "Kubernetes Production Cluster US-East",
            "EKS 1.29",
            "Primary production container orchestration cluster in us-east-1.",
            "team:cloud-infrastructure",
            ["Prod K8s East", "k8s-east"],
        ),
        (
            "infra:k8s-prod-cluster-us-west",
            "Kubernetes Production Cluster US-West",
            "EKS 1.29",
            "Secondary regional cluster in us-west-2 for warehouse operations.",
            "team:cloud-infrastructure",
            ["Prod K8s West", "k8s-west"],
        ),
        (
            "infra:kafka-event-bus",
            "Kafka Production Event Bus",
            "Apache Kafka 3.7",
            "Enterprise event backbone streaming order, inventory, and audit events.",
            "team:data-platform",
            ["Kafka Event Stream", "event-bus"],
        ),
        (
            "infra:pci-isolated-vpc",
            "PCI Isolated Secure VPC",
            "AWS Isolated VPC",
            "Hardened network segment dedicated to payment processing workloads.",
            "team:security-secops",
            ["PCI VPC", "secure-enclave"],
        ),
        (
            "infra:aws-rds-postgres-cluster",
            "AWS RDS PostgreSQL Multi-AZ Cluster",
            "AWS RDS Multi-AZ",
            "Managed PostgreSQL database instances with read replicas.",
            "team:data-platform",
            ["RDS Postgres", "aws-rds-cluster"],
        ),
        (
            "infra:aws-elasticache-redis",
            "AWS ElastiCache Redis Cluster",
            "AWS ElastiCache",
            "Managed in-memory Redis replication group for caching.",
            "team:data-platform",
            ["ElastiCache Redis", "redis-cluster"],
        ),
        (
            "infra:mongodb-atlas-cluster",
            "MongoDB Atlas Managed Cluster",
            "MongoDB Atlas",
            "Multi-region cloud MongoDB document store.",
            "team:catalog-search",
            ["Atlas Cluster", "mongo-atlas"],
        ),
        (
            "infra:aws-opensearch-cluster",
            "AWS OpenSearch Service Cluster",
            "AWS OpenSearch",
            "Managed Elasticsearch cluster for catalog query execution.",
            "team:catalog-search",
            ["OpenSearch Cluster", "opensearch-prod"],
        ),
        (
            "infra:clickhouse-analytics-cluster",
            "ClickHouse Analytics Cluster",
            "ClickHouse Managed",
            "Distributed columnar analytics database.",
            "team:data-platform",
            ["ClickHouse Cluster", "analytics-infra"],
        ),
        (
            "infra:aws-edge-cloudfront",
            "AWS CloudFront Edge CDN & WAF",
            "AWS CloudFront",
            "Edge delivery network with attached Web Application Firewall.",
            "team:security-secops",
            ["CloudFront CDN", "edge-waf"],
        ),
    ]

    for inf_id, inf_name, tech, desc, owner_team, aliases in infra_data:
        doc_id = f"doc-{inf_id.replace(':', '-')}"
        docs.append(
            {
                "document_id": doc_id,
                "source_uri": f"docs/infrastructure/{doc_id}.md",
                "title": f"Infrastructure Specification: {inf_name}",
                "doc_type": "infra_spec",
                "content": (
                    f"# Infrastructure Specification: {inf_name}\n\n"
                    f"**Canonical Identifier:** `{inf_id}`\n"
                    f"**Technology Stack:** {tech}\n"
                    f"**Owner Team (`OWNED_BY`):** `{owner_team}`\n"
                    f"**Known Aliases:** {', '.join(aliases)}\n\n"
                    f"## Infrastructure Role\n"
                    f"{desc}\n"
                ),
                "metadata": {
                    "entity_id": inf_id,
                    "entity_type": "Infrastructure",
                    "technology": tech,
                    "owner_team": owner_team,
                    "aliases": aliases,
                },
            }
        )

    # -------------------------------------------------------------------------
    # 6. REPOSITORIES (10 Documents)
    # -------------------------------------------------------------------------
    repos_data = [
        (
            "repo:checkout-service-repo",
            "checkout-service-repo",
            "team:checkout-engineering",
            "service:checkout-service",
            "github.com/aegis/checkout-service",
            ["checkout-repo"],
        ),
        (
            "repo:payment-gateway-repo",
            "payment-gateway-repo",
            "team:checkout-engineering",
            "service:payment-gateway",
            "github.com/aegis/payment-gateway",
            ["payment-repo", "pay-gw-repo"],
        ),
        (
            "repo:auth-service-repo",
            "auth-service-repo",
            "team:identity-access",
            "service:auth-service",
            "github.com/aegis/auth-service",
            ["auth-repo"],
        ),
        (
            "repo:inventory-service-repo",
            "inventory-service-repo",
            "team:inventory-fulfillment",
            "service:inventory-service",
            "github.com/aegis/inventory-service",
            ["inventory-repo"],
        ),
        (
            "repo:catalog-service-repo",
            "catalog-service-repo",
            "team:catalog-search",
            "service:catalog-service",
            "github.com/aegis/catalog-service",
            ["catalog-repo"],
        ),
        (
            "repo:notification-service-repo",
            "notification-service-repo",
            "team:notification-messaging",
            "service:notification-service",
            "github.com/aegis/notification-service",
            ["notifications-repo"],
        ),
        (
            "repo:billing-service-repo",
            "billing-service-repo",
            "team:billing-finance",
            "service:billing-service",
            "github.com/aegis/billing-service",
            ["billing-repo"],
        ),
        (
            "repo:data-pipeline-repo",
            "data-pipeline-repo",
            "team:data-platform",
            "service:data-pipeline-service",
            "github.com/aegis/data-pipeline",
            ["etl-repo"],
        ),
        (
            "repo:ticketing-service-repo",
            "ticketing-service-repo",
            "team:customer-support-eng",
            "service:ticketing-service",
            "github.com/aegis/ticketing-service",
            ["support-repo"],
        ),
        (
            "repo:security-tools-repo",
            "security-tools-repo",
            "team:security-secops",
            "service:audit-logger-service",
            "github.com/aegis/secops-tools",
            ["secops-repo"],
        ),
    ]

    for repo_id, repo_name, owner_team, hosted_svc, repo_url, aliases in repos_data:
        doc_id = f"doc-{repo_id.replace(':', '-')}"
        docs.append(
            {
                "document_id": doc_id,
                "source_uri": f"docs/repositories/{doc_id}.md",
                "title": f"Repository Manifest: {repo_name}",
                "doc_type": "repository_manifest",
                "content": (
                    f"# Repository Manifest: {repo_name}\n\n"
                    f"**Canonical Identifier:** `{repo_id}`\n"
                    f"**Owner Team (`OWNED_BY`):** `{owner_team}`\n"
                    f"**Stores Code For (`STORES_CODE`):** `{hosted_svc}`\n"
                    f"**Remote URL:** `{repo_url}`\n"
                    f"**Known Aliases:** {', '.join(aliases)}\n\n"
                    f"## Branching & CI/CD Pipeline\n"
                    f"Protected main branch with required GitHub Actions testing and SonarQube static analysis."
                ),
                "metadata": {
                    "entity_id": repo_id,
                    "entity_type": "Repository",
                    "owner_team": owner_team,
                    "service": hosted_svc,
                    "aliases": aliases,
                },
            }
        )

    # -------------------------------------------------------------------------
    # 7. VULNERABILITIES & CVES (10 Documents)
    # -------------------------------------------------------------------------
    cves_data = [
        (
            "vuln:cve-2024-21626",
            "CVE-2024-21626",
            "9.8 CRITICAL",
            "service:payment-gateway",
            "runc container breakout vulnerability in base images.",
            "Upgrade base images to runc 1.1.12.",
            ["CVE-2024-21626", "runc escape"],
        ),
        (
            "vuln:cve-2024-3094",
            "CVE-2024-3094",
            "10.0 CRITICAL",
            "service:auth-service",
            "Malicious backdoor in upstream XZ Utils compression library.",
            "Downgrade xz-utils and rebuild authentication containers.",
            ["CVE-2024-3094", "XZ Backdoor"],
        ),
        (
            "vuln:cve-2023-44487",
            "CVE-2023-44487",
            "7.5 HIGH",
            "service:checkout-service",
            "HTTP/2 Rapid Reset DDoS vulnerability in gRPC ingress.",
            "Apply Envoy proxy patch with HTTP/2 stream limit.",
            ["HTTP/2 Rapid Reset", "CVE-2023-44487"],
        ),
        (
            "vuln:cve-2024-28180",
            "CVE-2024-28180",
            "8.1 HIGH",
            "service:notification-service",
            "JOSE JSON Web Encryption decompression bomb vulnerability.",
            "Enforce strict payload decompression limits.",
            ["CVE-2024-28180"],
        ),
        (
            "vuln:cve-2023-4863",
            "CVE-2023-4863",
            "8.8 HIGH",
            "service:catalog-service",
            "Heap buffer overflow in libwebp during product image processing.",
            "Update libwebp to version 1.3.2.",
            ["libwebp overflow", "CVE-2023-4863"],
        ),
        (
            "vuln:cve-2024-23897",
            "CVE-2024-23897",
            "9.8 CRITICAL",
            "service:data-pipeline-service",
            "Arbitrary file read in Jenkins CLI parser.",
            "Disable CLI over WebSocket and upgrade Jenkins controller.",
            ["Jenkins CLI exploit", "CVE-2024-23897"],
        ),
        (
            "vuln:cve-2023-38545",
            "CVE-2023-38545",
            "9.8 CRITICAL",
            "service:sms-relay-service",
            "SOCKS5 heap buffer overflow in cURL library.",
            "Recompile container with libcurl 8.4.0.",
            ["curl SOCKS5 overflow", "CVE-2023-38545"],
        ),
        (
            "vuln:cve-2024-2961",
            "CVE-2024-2961",
            "7.5 HIGH",
            "service:billing-service",
            "glibc iconv buffer overflow in character conversion routines.",
            "Patch system glibc packages to 2.39-0ubuntu8.1.",
            ["glibc iconv CVE", "CVE-2024-2961"],
        ),
        (
            "vuln:cve-2023-50164",
            "CVE-2023-50164",
            "9.8 CRITICAL",
            "service:ticketing-service",
            "Remote code execution path traversal in file upload handlers.",
            "Upgrade file parsing libraries and sanitize file names.",
            ["Struts path traversal", "CVE-2023-50164"],
        ),
        (
            "vuln:cve-2024-24790",
            "CVE-2024-24790",
            "7.5 HIGH",
            "service:user-directory-service",
            "Go standard library net/netip unexpected IP address parsing.",
            "Recompile service binary with Go 1.22.4.",
            ["Go netip vulnerability", "CVE-2024-24790"],
        ),
    ]

    for cve_id, cve_title, severity, affected_svc, impact_desc, fix_desc, aliases in cves_data:
        doc_id = f"doc-{cve_id.replace(':', '-')}"
        docs.append(
            {
                "document_id": doc_id,
                "source_uri": f"docs/security/{doc_id}.md",
                "title": f"Security Advisory: {cve_title}",
                "doc_type": "security_advisory",
                "content": (
                    f"# Security Advisory: {cve_title}\n\n"
                    f"**Canonical Identifier:** `{cve_id}`\n"
                    f"**Severity:** {severity}\n"
                    f"**Impacted Service (`AFFECTED_BY`):** `{affected_svc}`\n"
                    f"**Advisory Aliases:** {', '.join(aliases)}\n\n"
                    f"## Threat Description\n"
                    f"{impact_desc}\n\n"
                    f"## Required Remediation\n"
                    f"{fix_desc}\n\n"
                    f"Security auditing and patch validation is supervised by `team:security-secops`."
                ),
                "metadata": {
                    "entity_id": cve_id,
                    "entity_type": "Vulnerability",
                    "severity": severity,
                    "affected_service": affected_svc,
                    "aliases": aliases,
                },
            }
        )

    # -------------------------------------------------------------------------
    # 8. ARCHITECTURE DECISION RECORDS (ADRs) (15 Documents)
    # -------------------------------------------------------------------------
    adrs_data = [
        (
            "ADR-101",
            "Adopt PostgreSQL with pgvector for Vector Search",
            "service:data-pipeline-service",
            "team:data-platform",
            "database:customer-orders-pg",
            "Accepted",
            "Standardized on PostgreSQL pgvector to reduce operational datastore overhead.",
        ),
        (
            "ADR-102",
            "Enforce Strict gRPC Protocols Between Checkout and Payment Gateway",
            "service:checkout-service",
            "team:checkout-engineering",
            "service:payment-gateway",
            "Accepted",
            "Replaced REST JSON with protobuf gRPC for lower latency payment processing.",
        ),
        (
            "ADR-103",
            "Isolate Payment Vault in PCI Enclave",
            "service:payment-gateway",
            "team:checkout-engineering",
            "database:payment-vault-pg",
            "Accepted",
            "Separated cardholder data into dedicated VPC (`infra:pci-isolated-vpc`).",
        ),
        (
            "ADR-104",
            "JWT Token Validation Strategy with Redis Caching",
            "service:auth-service",
            "team:identity-access",
            "database:auth-session-redis",
            "Accepted",
            "Cache public key certs and revoked JTI tokens in Redis.",
        ),
        (
            "ADR-105",
            "Asynchronous Inventory Reservations via Kafka",
            "service:inventory-service",
            "team:inventory-fulfillment",
            "infra:kafka-event-bus",
            "Accepted",
            "Publish out-of-stock events to Kafka for non-blocking catalog updates.",
        ),
        (
            "ADR-106",
            "Migrate Product Catalog Store to MongoDB Atlas",
            "service:catalog-service",
            "team:catalog-search",
            "database:catalog-mongodb",
            "Accepted",
            "Leverage flexible document schema for complex product variants.",
        ),
        (
            "ADR-107",
            "Dual-Store Invariant for Enterprise Knowledge Graph",
            "service:data-pipeline-service",
            "team:data-platform",
            "database:customer-orders-pg",
            "Accepted",
            "Enforce shared canonical UUID chunk_id across Neo4j and PostgreSQL.",
        ),
        (
            "ADR-108",
            "Centralize Audit Trail Persistence in S3",
            "service:audit-logger-service",
            "team:security-secops",
            "infra:kafka-event-bus",
            "Accepted",
            "Write append-only audit event streams directly to object storage.",
        ),
        (
            "ADR-109",
            "Zero Raw Cypher Generation Guardrail",
            "service:data-pipeline-service",
            "team:security-secops",
            "team:data-platform",
            "Accepted",
            "Mandate parameterized Cypher templates for 100% graph query safety.",
        ),
        (
            "ADR-110",
            "Implement Automatic Rate Limiting at Edge CloudFront",
            "service:waf-manager",
            "team:security-secops",
            "infra:aws-edge-cloudfront",
            "Accepted",
            "Block malicious scrapers before hitting Kubernetes ingress.",
        ),
        (
            "ADR-111",
            "Debezium CDC Pipeline for Real-time ClickHouse Analytics",
            "service:data-pipeline-service",
            "team:data-platform",
            "database:analytics-clickhouse",
            "Accepted",
            "Capture database changes without impacting production PostgreSQL OLTP performance.",
        ),
        (
            "ADR-112",
            "Twilio Carrier Fallback for SMS Delivery",
            "service:sms-relay-service",
            "team:notification-messaging",
            "service:notification-service",
            "Accepted",
            "Add multi-carrier routing to guarantee 99.99% SMS delivery SLAs.",
        ),
        (
            "ADR-113",
            "Double-Entry Ledger Architecture for Subscription Billing",
            "service:billing-service",
            "team:billing-finance",
            "database:financial-ledger-pg",
            "Accepted",
            "Enforce financial invariant that sum of debits equals credits.",
        ),
        (
            "ADR-114",
            "Automate EKS Pod Autoscaling with KEDA",
            "service:cluster-orchestrator",
            "team:cloud-infrastructure",
            "infra:k8s-prod-cluster-us-east",
            "Accepted",
            "Autoscale worker pods dynamically based on Kafka lag metrics.",
        ),
        (
            "ADR-115",
            "Customer Support CRM Bidirectional Sync Protocol",
            "service:crm-sync-service",
            "team:customer-support-eng",
            "service:ticketing-service",
            "Accepted",
            "Establish webhook listeners to keep customer contacts synchronized.",
        ),
    ]

    for adr_num, title, related_svc, author_team, related_target, status, rationale in adrs_data:
        doc_id = f"doc-adr-{adr_num.lower()}"
        docs.append(
            {
                "document_id": doc_id,
                "source_uri": f"docs/architecture-decisions/{doc_id}.md",
                "title": f"Architecture Decision Record: {adr_num} — {title}",
                "doc_type": "architecture_decision_record",
                "content": (
                    f"# Architecture Decision Record: {adr_num} — {title}\n\n"
                    f"**Status:** {status}\n"
                    f"**Authoring Team (`OWNED_BY`):** `{author_team}`\n"
                    f"**Primary System (`DEPENDS_ON`):** `{related_svc}`\n"
                    f"**Impacted Component:** `{related_target}`\n\n"
                    f"## Context & Rationale\n"
                    f"{rationale}\n\n"
                    f"## Consequences & Invariants\n"
                    f"All systems depending on `{related_svc}` must adhere strictly to this architectural contract."
                ),
                "metadata": {
                    "entity_id": f"adr:{adr_num.lower()}",
                    "entity_type": "ArchitectureDecision",
                    "author_team": author_team,
                    "related_service": related_svc,
                    "status": status,
                },
            }
        )

    # -------------------------------------------------------------------------
    # 9. POST-MORTEM & INCIDENT REPORTS (15 Documents)
    # -------------------------------------------------------------------------
    incidents_data = [
        (
            "INC-2024-01",
            "Payment Gateway Timeout Cascading Failure",
            "service:payment-gateway",
            "team:checkout-engineering",
            "service:checkout-service",
            "database:payment-vault-pg",
            "High database connection pool exhaustion in payment-vault-pg caused checkout timeouts.",
        ),
        (
            "INC-2024-02",
            "Auth Service Memory Leak During Peak Load",
            "service:auth-service",
            "team:identity-access",
            "service:checkout-service",
            "database:auth-session-redis",
            "Unbounded cache growth in auth-session-redis triggered pod OOMKills.",
        ),
        (
            "INC-2024-03",
            "Inventory Over-Reservation Race Condition",
            "service:inventory-service",
            "team:inventory-fulfillment",
            "service:cart-service",
            "database:inventory-pg",
            "Missing row-level locks on inventory-pg permitted duplicate reservation tokens.",
        ),
        (
            "INC-2024-04",
            "Catalog Search Degradation via Malformed Queries",
            "service:catalog-service",
            "team:catalog-search",
            "service:search-indexer",
            "database:search-elasticsearch",
            "Deep un-paginated Elasticsearch queries spiked cluster CPU to 100%.",
        ),
        (
            "INC-2024-05",
            "Kafka Broker Partition Imbalance",
            "service:data-pipeline-service",
            "team:data-platform",
            "infra:kafka-event-bus",
            "database:customer-orders-pg",
            "Uneven partition hashing caused order processing lag of 45 minutes.",
        ),
        (
            "INC-2024-06",
            "SMS Gateway Delivery Outage",
            "service:sms-relay-service",
            "team:notification-messaging",
            "service:notification-service",
            "service:auth-service",
            "Carrier upstream certificate expiry blocked 2FA text delivery.",
        ),
        (
            "INC-2024-07",
            "Invoice Rounding Discrepancy",
            "service:billing-service",
            "team:billing-finance",
            "service:tax-calculator-service",
            "database:financial-ledger-pg",
            "Floating point tax calculation in tax-calculator-service created cent mismatches in ledger.",
        ),
        (
            "INC-2024-08",
            "Kubernetes Node Group Eviction Storm",
            "service:cluster-orchestrator",
            "team:cloud-infrastructure",
            "infra:k8s-prod-cluster-us-east",
            "service:checkout-service",
            "Subnet IP exhaustion in us-east VPC prevented new node group attachments.",
        ),
        (
            "INC-2024-09",
            "Ticket Sync Deadlock",
            "service:ticketing-service",
            "team:customer-support-eng",
            "service:crm-sync-service",
            "database:crm-ticketing-pg",
            "Simultaneous CRM webhook updates triggered Postgres deadlocks in crm-ticketing-pg.",
        ),
        (
            "INC-2024-10",
            "WAF Misconfiguration Blocking Checkout API",
            "service:waf-manager",
            "team:security-secops",
            "api:create-order-v2",
            "service:checkout-service",
            "Overly aggressive regex rule in WAF flagged valid order payload JSON.",
        ),
        (
            "INC-2024-11",
            "Audit Log Buffer Overflow",
            "service:audit-logger-service",
            "team:security-secops",
            "infra:kafka-event-bus",
            "service:payment-gateway",
            "Spike in payment audits saturated local logger ring buffer.",
        ),
        (
            "INC-2024-12",
            "Warehouse Robotics Sync Disconnect",
            "service:warehouse-service",
            "team:inventory-fulfillment",
            "service:inventory-service",
            "infra:k8s-prod-cluster-us-west",
            "Cross-region VPN tunnel flap caused warehouse queue disconnection.",
        ),
        (
            "INC-2024-13",
            "MongoDB Atlas Primary Election Latency",
            "service:catalog-service",
            "team:catalog-search",
            "database:catalog-mongodb",
            "service:cart-service",
            "Atlas primary node failover took 32 seconds, dropping catalog reads.",
        ),
        (
            "INC-2024-14",
            "ClickHouse Ingestion Lag During Flash Sale",
            "service:data-pipeline-service",
            "team:data-platform",
            "database:analytics-clickhouse",
            "infra:kafka-event-bus",
            "ETL ingestion worker backlog reached 2.4 million pending events.",
        ),
        (
            "INC-2024-15",
            "CVE-2024-3094 Proactive Isolation and Quarantine",
            "service:auth-service",
            "team:security-secops",
            "vuln:cve-2024-3094",
            "repo:auth-service-repo",
            "Emergency deployment rolled back vulnerable XZ package across identity clusters.",
        ),
    ]

    for (
        inc_num,
        title,
        root_svc,
        responding_team,
        impacted_comp,
        related_res,
        summary,
    ) in incidents_data:
        doc_id = f"doc-incident-{inc_num.lower()}"
        docs.append(
            {
                "document_id": doc_id,
                "source_uri": f"docs/post-mortems/{doc_id}.md",
                "title": f"Incident Post-Mortem: {inc_num} — {title}",
                "doc_type": "post_mortem",
                "content": (
                    f"# Incident Post-Mortem: {inc_num} — {title}\n\n"
                    f"**Primary Impacted Service (`DEPENDS_ON`):** `{root_svc}`\n"
                    f"**Responding Incident Team (`OWNED_BY`):** `{responding_team}`\n"
                    f"**Upstream/Downstream Impact:** `{impacted_comp}`\n"
                    f"**Related Resource:** `{related_res}`\n\n"
                    f"## Executive Summary & Root Cause\n"
                    f"{summary}\n\n"
                    f"## Action Items & Preventative Measures\n"
                    f"1. Update architectural timeout limits and circuit breakers for `{root_svc}`.\n"
                    f"2. Add telemetry alerts to `service:monitoring-collector` managed by `team:cloud-infrastructure`."
                ),
                "metadata": {
                    "entity_id": f"incident:{inc_num.lower()}",
                    "entity_type": "Incident",
                    "responding_team": responding_team,
                    "root_service": root_svc,
                    "impacted_component": impacted_comp,
                },
            }
        )

    # -------------------------------------------------------------------------
    # 10. RUNBOOKS & OPERATIONAL GUIDES (15 Documents)
    # -------------------------------------------------------------------------
    runbooks_data = [
        (
            "RB-01",
            "Payment Gateway Failover Runbook",
            "service:payment-gateway",
            "team:checkout-engineering",
            "Steps to switch primary payment processing to standby backup gateway.",
        ),
        (
            "RB-02",
            "Emergency Auth Service Secret Rotation",
            "service:auth-service",
            "team:identity-access",
            "Runbook for rotating JWT signing RSA keys in Redis without customer logout.",
        ),
        (
            "RB-03",
            "Inventory Reconciliation and Drift Repair",
            "service:inventory-service",
            "team:inventory-fulfillment",
            "Nightly batch script execution to align physical stock with PostgreSQL counts.",
        ),
        (
            "RB-04",
            "Elasticsearch Cluster Reindexing Procedure",
            "service:catalog-service",
            "team:catalog-search",
            "Safe zero-downtime blue/green reindexing of catalog search indices.",
        ),
        (
            "RB-05",
            "PostgreSQL Database Failover and PITR Recovery",
            "database:customer-orders-pg",
            "team:data-platform",
            "Standard operating procedure for promoting AWS RDS read replicas.",
        ),
        (
            "RB-06",
            "Kafka Topic Compaction and Re-partitioning",
            "infra:kafka-event-bus",
            "team:data-platform",
            "Procedure for scaling Kafka partitions without consumer rebalance storms.",
        ),
        (
            "RB-07",
            "CVE Patch Deployment and Vulnerability Scanning",
            "service:audit-logger-service",
            "team:security-secops",
            "Triage and deployment protocol for addressing High/Critical CVE alerts.",
        ),
        (
            "RB-08",
            "Billing Ledger Month-End Settlement Runbook",
            "service:billing-service",
            "team:billing-finance",
            "Financial month-end ledger closing and revenue audit procedures.",
        ),
        (
            "RB-09",
            "Kubernetes Cluster Emergency Node Draining",
            "infra:k8s-prod-cluster-us-east",
            "team:cloud-infrastructure",
            "Safely cordon and drain Kubernetes worker nodes during AWS outages.",
        ),
        (
            "RB-10",
            "Support Ticket Escalation to Engineering",
            "service:ticketing-service",
            "team:customer-support-eng",
            "Triage pathway for escalating critical customer bugs to Checkout Engineering.",
        ),
        (
            "RB-11",
            "WAF Custom Rate Limiting Rule Deployment",
            "service:waf-manager",
            "team:security-secops",
            "Deploying custom CloudFront WAF rules to block DDoS botnets.",
        ),
        (
            "RB-12",
            "ClickHouse Historical Partition Archiving",
            "database:analytics-clickhouse",
            "team:data-platform",
            "Compressing and cold-tiering historical analytics tables to S3 storage.",
        ),
        (
            "RB-13",
            "SMS Gateway Provider Outage Switchover",
            "service:sms-relay-service",
            "team:notification-messaging",
            "Toggling secondary SMS provider routing during carrier incidents.",
        ),
        (
            "RB-14",
            "Product Catalog Cache Invalidation Procedure",
            "database:catalog-mongodb",
            "team:catalog-search",
            "Purging stale product pricing from edge CDN and Redis caches.",
        ),
        (
            "RB-15",
            "Kubernetes Cluster Zero-Downtime EKS Upgrade",
            "infra:k8s-prod-cluster-us-west",
            "team:cloud-infrastructure",
            "Step-by-step rolling upgrade of Kubernetes control planes and node groups.",
        ),
    ]

    for rb_num, title, target_entity, owner_team, description in runbooks_data:
        doc_id = f"doc-runbook-{rb_num.lower()}"
        docs.append(
            {
                "document_id": doc_id,
                "source_uri": f"docs/runbooks/{doc_id}.md",
                "title": f"Operational Runbook: {rb_num} — {title}",
                "doc_type": "operational_runbook",
                "content": (
                    f"# Operational Runbook: {rb_num} — {title}\n\n"
                    f"**Governing Team (`OWNED_BY`):** `{owner_team}`\n"
                    f"**Target System/Resource:** `{target_entity}`\n\n"
                    f"## Purpose & Execution Steps\n"
                    f"{description}\n\n"
                    f"## Safety Verification\n"
                    f"Ensure all actions are logged in `service:audit-logger-service` and verified by on-call engineers."
                ),
                "metadata": {
                    "entity_id": f"runbook:{rb_num.lower()}",
                    "entity_type": "Runbook",
                    "owner_team": owner_team,
                    "target_entity": target_entity,
                },
            }
        )

    # -------------------------------------------------------------------------
    # 11. CROSS-FUNCTIONAL INTEGRATION SPECS (15 Documents)
    # -------------------------------------------------------------------------
    integration_data = [
        (
            "INT-01",
            "Checkout to Payment Flow Integration",
            "service:checkout-service",
            "service:payment-gateway",
            "team:checkout-engineering",
            "Describes gRPC handshake and idempotent token handling between Checkout and Payment Gateway.",
        ),
        (
            "INT-02",
            "Payment Gateway to Financial Ledger Sync",
            "service:payment-gateway",
            "service:billing-service",
            "team:billing-finance",
            "Details event-driven journal posting of successful payment captures.",
        ),
        (
            "INT-03",
            "Checkout to Inventory Reservation Handshake",
            "service:checkout-service",
            "service:inventory-service",
            "team:inventory-fulfillment",
            "Synchronous lock acquisition preventing cart race conditions during sales.",
        ),
        (
            "INT-04",
            "Authentication Token Validation Pipeline",
            "service:checkout-service",
            "service:auth-service",
            "team:identity-access",
            "Standard OAuth2 Bearer token inspection across public API gateways.",
        ),
        (
            "INT-05",
            "Order Placement Notification Dispatch Flow",
            "service:checkout-service",
            "service:notification-service",
            "team:notification-messaging",
            "Event-driven receipt email and push notification dispatch upon order completion.",
        ),
        (
            "INT-06",
            "Support Portal Order History Integration",
            "service:ticketing-service",
            "service:checkout-service",
            "team:customer-support-eng",
            "Secure internal API bridging CRM tickets with customer order line items.",
        ),
        (
            "INT-07",
            "Search Index Ingestion Streaming Spec",
            "service:search-indexer",
            "service:catalog-service",
            "team:catalog-search",
            "CDC pipeline synchronizing MongoDB catalog updates to Elasticsearch cluster.",
        ),
        (
            "INT-08",
            "Transactional Database CDC to Analytics Warehouse",
            "service:data-pipeline-service",
            "database:customer-orders-pg",
            "team:data-platform",
            "Debezium streaming of PostgreSQL WAL logs into ClickHouse analytics tables.",
        ),
        (
            "INT-09",
            "PCI Enclave Ingress and Tokenization Protocol",
            "service:payment-gateway",
            "infra:pci-isolated-vpc",
            "team:security-secops",
            "Network isolation and mTLS mutual authentication protocol for payment processing.",
        ),
        (
            "INT-10",
            "Warehouse Robotic Dispatch Queue Protocol",
            "service:inventory-service",
            "service:warehouse-service",
            "team:inventory-fulfillment",
            "Queue protocol forwarding confirmed stock allocations to automated warehouse cranes.",
        ),
        (
            "INT-11",
            "Edge WAF Rate Limiting and Token Bucket Policy",
            "service:waf-manager",
            "api:create-order-v2",
            "team:security-secops",
            "Configures 100 req/min token bucket limit on order creation API.",
        ),
        (
            "INT-12",
            "Tax Calculation Integration Protocol",
            "service:billing-service",
            "service:tax-calculator-service",
            "team:billing-finance",
            "Synchronous REST exchange calculating multi-jurisdiction tax totals.",
        ),
        (
            "INT-13",
            "User Profile Account Verification Protocol",
            "service:user-directory-service",
            "service:sms-relay-service",
            "team:identity-access",
            "SMS 2FA token generation and delivery validation flow for user sign-ups.",
        ),
        (
            "INT-14",
            "Audit Logger Security Pipeline Integration",
            "service:audit-logger-service",
            "infra:kafka-event-bus",
            "team:security-secops",
            "Zero-loss Kafka audit topic configuration with consumer replication.",
        ),
        (
            "INT-15",
            "Kubernetes Pod Autoscaling Kafka Consumer Metric Flow",
            "service:cluster-orchestrator",
            "infra:kafka-event-bus",
            "team:cloud-infrastructure",
            "Custom metric adapter triggering horizontal pod autoscaling from Kafka consumer lag.",
        ),
    ]

    for int_num, title, source_comp, target_comp, lead_team, desc in integration_data:
        doc_id = f"doc-integration-{int_num.lower()}"
        docs.append(
            {
                "document_id": doc_id,
                "source_uri": f"docs/integrations/{doc_id}.md",
                "title": f"Integration Specification: {int_num} — {title}",
                "doc_type": "integration_spec",
                "content": (
                    f"# Integration Specification: {int_num} — {title}\n\n"
                    f"**Lead Team (`OWNED_BY`):** `{lead_team}`\n"
                    f"**Source Component (`DEPENDS_ON`):** `{source_comp}`\n"
                    f"**Target Component:** `{target_comp}`\n\n"
                    f"## Interface Contract\n"
                    f"{desc}\n\n"
                    f"## Resilience & Circuit Breaking\n"
                    f"Circuit breakers must trip after 3 consecutive 500ms timeouts."
                ),
                "metadata": {
                    "entity_id": f"integration:{int_num.lower()}",
                    "entity_type": "Integration",
                    "lead_team": lead_team,
                    "source_component": source_comp,
                    "target_component": target_comp,
                },
            }
        )

    return docs


def write_corpus_jsonl(output_path: Path) -> int:
    """Generate and write the complete synthetic enterprise corpus to JSONL."""
    docs = build_corpus_documents()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            line = json.dumps(doc, ensure_ascii=False)
            f.write(line + "\n")
    return len(docs)


if __name__ == "__main__":
    out_file = Path("data/raw/corpus.jsonl")
    count = write_corpus_jsonl(out_file)
    print(f"Successfully generated {count} documents in {out_file.resolve()}")
