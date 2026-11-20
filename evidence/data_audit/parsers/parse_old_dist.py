import sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from parse_crops import dist as _dist
def dist(n):
    k = n.lower().replace(' ', '').replace('.', '')
    if 'nawabgonj' in k or 'nawbgonj' in k: return 'Chapainawabganj'
    if k == 'noagaon': return 'Naogaon'
    return _dist(n)
