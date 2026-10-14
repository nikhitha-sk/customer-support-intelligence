# Software Troubleshooting: Crashes, Bugs and Performance

Queue: Technical Support

## Standard troubleshooting checklist
1. Confirm the product version and operating system.
2. Check whether the issue started after an update; if so, note the previous and current versions.
3. Ask the customer to restart the application and clear the local cache.
4. Collect the application log (Help > Export diagnostics) and any error codes.
5. Try to reproduce the issue in a clean environment.

## Crashes after updates
If crashes began after an update, the customer can roll back to the previous stable version from the downloads page while engineering investigates. Record the crash signature in the ticket so duplicate reports can be linked to the same bug.

## Slow performance
Common causes are large local caches, outdated graphics drivers, low disk space (below 10%), or syncing very large projects. For web products, ask the customer to test in a private browser window to rule out extensions.

## Integration and API errors
For API errors, collect the request ID, timestamp, HTTP status code and endpoint. 401/403 errors usually indicate expired or under-scoped API keys. 429 errors mean rate limits were exceeded; the standard limit is 100 requests per minute per key, and enterprise plans can request higher limits.

## Escalating bugs
Escalate to engineering when the issue is reproducible, affects multiple customers, or blocks a core workflow with no workaround. Include reproduction steps, logs and impact in the escalation.
