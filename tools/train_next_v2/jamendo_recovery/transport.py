"""Retry only observed TLS EOF during connection setup, retaining all range gates."""
import re
import time
import requests


class BoundedTLSSession:
    def __init__(self, budget, log, factory=requests.Session, sleep=time.sleep):
        self.budget, self.log, self.factory, self.sleep = budget, log, factory, sleep
        self.session = factory()

    def get(self, url, **kwargs):
        if kwargs.get('verify', True) is not True:
            raise ValueError('TLS verification cannot be disabled')
        match = re.fullmatch(r'bytes=(\d+)-(\d+)', kwargs['headers']['Range'])
        if not match:
            raise ValueError('An exact byte range is required')
        first, last = map(int, match.groups())
        if not 0 <= first <= last:
            raise ValueError('Invalid range')
        for attempt in range(1, 4):
            try:
                return self.session.get(url, **kwargs)
            except requests.exceptions.SSLError as exc:
                # Certificate errors, HTTP errors, source changes and body failures
                # do not enter this observed connection-EOF recovery path.
                recoverable = 'UNEXPECTED_EOF_WHILE_READING' in str(exc)
                self.log(dict(event='connection_TLS_error', url=url, byte_range=kwargs['headers']['Range'],
                    attempt=attempt, recoverable=recoverable, error=str(exc)))
                if not recoverable or attempt == 3:
                    raise
                self.budget.reserve(last-first+1)  # original request already reserved by ArchiveRanges
                self.session.close()
                self.sleep(2 if attempt == 1 else 5)
                self.session = self.factory()

    def close(self):
        self.session.close()
