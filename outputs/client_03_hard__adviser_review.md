# Internal Reconciliation Review

## Reconciliation Review — internal, not for client distribution

Client: Robert Fletcher and Jean Fletcher

### Source conflicts (which source we trusted, and why)

- H-GIA-JF.value: chose 38000 @ 2026-05-16 (observed) over 30000 @ 2026-03-10 (db snapshot) (most-recent-valuation-wins)

### Value provenance

| Account | Value | Source | In scope |
|---|---|---|---|
| H-ISA-R | £70,000 | system of record (db) | yes |
| H-GIA-JF | £38,000 | fresher observed value | yes |
| H-ISA-JE | £66,000 | system of record (db) | yes |
| H-CASH-JE | — | system of record (db) | no |

### Outstanding flags for a human to finalise

- [FLAG: platform charge — ongoing platform charge to confirm]
- [FLAG: advice charge — ongoing advice charge to confirm]
- [FLAG: capital gains tax — liability on the disposal to be confirmed by adviser]
