#!/usr/bin/env python3

from pprint import pprint

import requests
import keyring


url = 'https://agency.f4.htw-berlin.de/dt'
auth = (
  keyring.get_password('red', 'agency_username'),
  keyring.get_password('red', 'agency_password')
)

r = requests.get(
  f'{url}/batches?limit=2',
  auth=auth
)

# r = requests.delete(
#   f'{url}/batches/5ede8ade821b4f33cf9a8bb8',
#   auth=auth
# )

r.raise_for_status()
batches = r.json()
pprint(batches)