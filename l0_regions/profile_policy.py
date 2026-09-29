"""Explicit research use of uncalibrated bounds, never an integrity bypass."""
POLICIES=('strict','research-report')


def validate_policy(policy):
    if policy not in POLICIES:raise ValueError('Unknown partition profile policy')
    return policy


def allow_profile(policy,debug):
    validate_policy(policy)
    if type(debug)!=bool:raise ValueError('Explicit DEBUG identity required')
    return debug or policy=='research-report'


def validate_cache_policy(meta,policy,debug):
    validate_policy(policy)
    if meta.get('profile_policy','strict')!=policy:raise ValueError('Explicit cache/profile policy mismatch')
    if meta['debug']!=debug:raise ValueError('Cache DEBUG mode mismatch')
    entries=[e for rows in meta['partitions'].values() for e in rows]
    if not entries or any(type(e['profile_exceeded'])!=bool for e in entries):raise ValueError('Profile evidence missing')
    failures=sum(e['profile_exceeded'] for e in entries)
    if type(meta['admission_failures'])!=int or failures!=meta['admission_failures']:raise ValueError('Profile violation count mismatch')
    strict=not debug and not failures and policy=='strict'
    research=not debug and policy=='research-report'
    if meta['full_training_admitted']!=strict or meta.get('research_training_admitted',False)!=research:
        raise ValueError('Profile admission classification mismatch')
    if failures and not allow_profile(policy,debug):raise ValueError('Initial profile rejected under strict policy')
