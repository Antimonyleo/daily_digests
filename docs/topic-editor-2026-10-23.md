# Topic editor refinement

The default topic editor now asks for a name and its inline priority. The old
Matching & terminology section, per-topic usefulness prompt and manual wording
variant button made a small edit look like a separate configuration task.

- Other names remain optional, collapsed and entered one per line. Existing
  semicolon-separated values still parse. Aliases continue to feed retrieval.
- The overall research goal remains the normal place to describe useful work.
  Existing per-topic descriptions are preserved, and remain editable under More
  about this topic when already present. New topics have no extra focus blank.
- PubMed spelling checks run after 700 ms without typing, on blur or on Done.
  Stored topics are not sent on page load. IME composition is allowed to finish.
- Only the topic name reaches the existing CSRF-protected spelling endpoint.
  Queries are serialized and spaced, with in-page result reuse. No profile,
  description, credentials or reading history accompanies them.
- Suggestions appear below the editing controls, including after Done. They do
  not replace the topic or save the desk. Use suggestion and Keep wording retain
  reader control; no-result checks stay quiet, and failures offer Retry.
- Editing again, cancelling, removing or replacing a row invalidates pending
  results. The existing server checks still protect acronyms and identifiers.
  Accepted corrections retain the original facet identity and untouched weights.

Focused Chrome 155 verification: 32 Settings cases passed, including debounce,
IME, blur/Done, acceptance/dismissal, failure/retry, stale replies, metadata,
legacy weights, and 1280/390/320 px layout. The new behavior failed before the
change. Native Settings/spelling/personalization tests passed 79 cases, and the
reading UI suite passed 12 checks. Two tests for the removed wording-variant
control were retired; its replacement behavior is exercised through real DOM
and mocked requests. Full static-browser acceptance and final CI remain release
gates, including native spelling identifier protections.
