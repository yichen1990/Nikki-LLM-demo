# LLM Security Notes

## Common Risks

- Prompt injection attacks
- Data exfiltration attempts
- Retrieval poisoning
- Hallucinated citations
- Structured output violations

## Prompt Injection Pattern

Indicators include:

- "Ignore previous instructions"
- "Reveal system prompt"
- "Override policy constraints"
- Requests for hidden configuration

## Mitigation Strategies

- Intent routing before generation
- Risk-scored triage layer
- Deterministic enforcement policies
- Structured JSON validation
- Logging and evaluation tracking

## Governance Recommendation

LLM systems deployed in academic environments should implement layered defenses rather than relying solely on alignment.
