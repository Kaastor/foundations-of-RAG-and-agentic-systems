---
doc_id: RB-X12-STAGING-ROTATE
title: X12 Staging Key Rotation Runbook
doc_type: runbook
product: X12
region: global
effective_date: 2025-02-15
version: 2025.02
authority: approved
status: active
disclosure: internal
allowed_roles: field_support,engineering,compliance
tags: x12,runbook,staging,rollback,key-rotation
references: 
trust: trusted
---
# Key rotation
Use the staging rotation checklist for sandbox and staging environments.

# Rollback steps
For staging, rollback requires these steps.
1. Freeze new token issuance.
2. Re-enable the previous staging API key.
3. Re-run the health check job.
4. Record the rollback in release checklist RC-22.

# Procedure alias
Internal teams sometimes call this procedure credential recovery after rotation.
