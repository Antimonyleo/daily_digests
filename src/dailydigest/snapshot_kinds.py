"""Source kinds the browser deployment's daily public snapshot carries.

Kept free of imports so build tools can read it before settings are loaded.
"""

# Public funding and event listings that are identical for every reader: they
# take no profile and ignore the look-back window. The browser deployment's
# daily snapshot carries them, so a site that changes or drops its CORS headers
# does not break browser brews; readers fall back to a direct fetch when the
# snapshot record is missing, failed or stale. Sources marked ``local_only``
# are still never collected.
SNAPSHOT_PAGE_KINDS = frozenset({
    "ukri_funding", "snsf_funding", "embl_events", "embo_events", "isscr_events",
    "keystone_events", "cshl_events", "nature_conferences", "wellcome_cs_events",
    "crs_events", "febs_events", "eci_events", "faseb_src_events",
})
